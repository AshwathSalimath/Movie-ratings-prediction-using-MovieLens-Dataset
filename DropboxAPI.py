from os.path import join
from os.path import isdir
from zipfile import ZipFile
from os import makedirs
import os
from urllib.request import urlretrieve


def unzip(zipFileName, destPath):
    """ Unzip ZIPFILENAME and store in DESTPATH. """

    filePath = join(destPath, zipFileName)
    zipFile = ZipFile(filePath, "r")
    zipFile.extractall(filePath.rsplit(".", 1)[0])
    """for dir in zipFile.printdir():
        print("", dir)
        if (zipFile.is_zipfile(dir)):
            #filePath = join(zipFile, dir)
            #zipFile = ZipFile(filePath, "r")
            dir.extractall(filePath.rsplit(".", 1)[0])
            dir.close()"""
    zipFile.close()


def fetchData(url, zipFileName, destPath):
    """ Fetch data from URL, unzip, and store in DESTPATH with name ZIPFILENAME. """

    filePath = join(destPath, zipFileName)

    # Create directory, if not present.
    if not isdir(destPath):
        makedirs(destPath)

    # Fetch data, unzip and store.
    urlretrieve(url, filePath)
    unzip(zipFileName, destPath)

url ='https://www.dropbox.com/sh/euppz607r6gsen2/AAAQCu8KjT7Ii1R60W2-Bm1Ua/MovieLens%20(Movie%20Ratings)?dl=1'
zipFileName = 'MovieLens (Movie Ratings).zip'
destPath = os.getcwd()
fetchData(url, zipFileName, destPath)