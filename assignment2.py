# Junjie Zhang, 9/19/2021

from __future__ import print_function

import sys
import re
import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.session import SparkSession
from operator import add


if __name__ == "__main__":

    # define a function to build up an array
    def buildArray(listOfIndices):
        returnVal = np.zeros(20000)

        for index in listOfIndices:
            returnVal[index] += 1

        mysum = np.sum(returnVal)
        returnVal = np.divide(returnVal, mysum)

        return returnVal

    sc = SparkContext(appName="KNN", conf=SparkConf().set('spark.driver.memory', '24g').set('spark.executor.memory', '12g'))
    spark = SparkSession(sc)

    # set up file path of the data set
    wikiPagesFile = sys.argv[1]
    wikiCategoryFile = sys.argv[2]

    # Read two files into RDDs

    # wiki pages
    wikiPages = sc.textFile(wikiPagesFile)

    # wiki category files
    wikiCategoryLinks = sc.textFile(wikiCategoryFile)
    wikiCats = wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))

    # get the total number of wikiPage file
    numberOfDocs = wikiPages.count()

    # Each entry in validLines will be a line from the text file, and transform it into a set of (docID, text) pairs
    validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('url="')-2], x[x.index('">') + 2:][:-6]))

    # split the text in each (docID, text) pair into a list of words
    # regular expression here to make sure that the program does not break down on some of the documents
    regex = re.compile('[^a-zA-Z]')

    # remove all non letter characters
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    # get the top 20,000 words, and sorted by frequency
    allWords = keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x, 1))
    allCounts = allWords.reduceByKey(add)
    topWords = allCounts.top(20000, lambda x: x[1])

    # create a RDD that has a set of (word, dictNum) pairs
    # transfer the top words to rdd, located base on the frequency in top words
    topWordsK = sc.parallelize(range(20000))
    dictionary = topWordsK.map(lambda x: (topWords[x][0], x))

    # create a RDD that has (docID, ['word1', 'word2', ...]
    allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # join and link to get a set of ("word1", (dictionaryPos, docID))
    allDictionaryWords = dictionary.join(allWordsWithDocID)

    # drop the actual word itself to get a set of (docID, dictionaryPos)
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

    # get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...])
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

    # converts the dictionary positions to a bag-of-words numpy array
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

    # create a new array in where every entry is either 1 or 0, 1 is occur while, 0 is not
    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], (x[1] > 0).astype(int)))

    # add up all of those arrays into a single array,
    dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

    # Create an array of 20,000 entries, each entry with the value numberOfDocs
    multiplier = np.full(20000, numberOfDocs)

    # Get the version of dfArray where the i^th entry is the inverse-document
    idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))

    # Get the version of dfArray where the i^th entry is the inverse-document frequency for the i^th word in the corpus
    allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

    print(allDocsAsNumpyArrays.take(3))
    print(allDocsAsNumpyArraysTFidf.take(2))

    # join allDocsAsNumpyArraysTFidf with categories, and map it leave the wikipageID only
    featuresRDD = wikiCats.join(allDocsAsNumpyArraysTFidf).map(lambda x: (x[1][0], x[1][1]))

    # Cache the data for KNN
    featuresRDD.cache()
    featuresRDD.take(10)

    # function that returns the prediction for the label of a string, using a kNN algorithm
    def getPrediction(textInput, k):
        # Create an RDD out of the textIput
        myDoc = sc.parallelize(('', textInput))
        # Flat map the text to (word, 1) pair for each word in the doc
        wordsInThatDoc = myDoc.flatMap(lambda x: ((j, 1) for j in regex.sub(' ', x).lower().split()))
        # get a set of (word, (dictionaryPos, 1)) pairs
        allDictionaryWordsInThatDoc = dictionary.join(wordsInThatDoc).map(lambda x: (x[1][1], x[1][0])).groupByKey()
        # Get tf array for the input string
        myArray = buildArray(allDictionaryWordsInThatDoc.top(1)[0][1])
        # Get the tf * idf array for the input string
        myArray = np.multiply(myArray, idfArray)
        # Get the distance from the input text string to all database documents
        distances = featuresRDD.map(lambda x: (x[0], np.dot(x[1], myArray)))
        # get the top k distances
        topK = distances.top(k, lambda x: x[1])
        # transform the top k distances into a set of (docID, 1)
        docIDRepresented = sc.parallelize(topK).map(lambda x: (x[0], 1))
        # get the count of the number of times this document ID appeared in the top k
        numTimes = docIDRepresented.reduceByKey(add)

        # Return the top 1 of them
        return numTimes.top(k, lambda x: x[1])


    print(getPrediction('Sport Basketball Volleyball Soccer', 10))
    print(getPrediction('What is the capital city of Australia?', 10))
    print(getPrediction('How many goals Vancouver score last year?', 10))

    # group the categories, and count
    cat_count = featuresRDD.map(lambda x: (x[0], 1)).reduceByKey(add)

    # conver rdd to dataframe and return the max, avg and standard deviation of counts
    columns = ['Category', 'Counts']
    df = cat_count.toDF(columns)
    max_count = df.agg({'Counts': 'max'}).collect()[0]
    avg_count = df.agg({'Counts': 'mean'}).collect()[0]
    std_count = df.agg({'Counts': 'stddev'}).collect()[0]

    print_out = sc.parallelize([(f'Max number of wikipedia categories of wilipedia pages is {max_count}',
          f'Average number of wikipedia categories of wilipedia pages is {avg_count}',
          f'Standard Deviation of wikipedia categories of wilipedia pages is {std_count}')]).coalesce(1)

    print_out.saveAsTextFile(sys.argv[3])

    # top 10 mostly used wikipedia categories
    output_2 = df.orderBy('Counts', ascending = False).limit(10).rdd

    output_2.saveAsTextFile(sys.argv[4])

    sc.stop()



