# Example Python program to upload a file to an FTP server

# in binary mode

from ftplib import FTP

 

# Create an FTP object and connect to the server

# as anonymous user

ftpObject = FTP(host="192.168.1.57");

print(ftpObject.getwelcome());

 

# Login to the server

ftpResponseMessage = ftpObject.login('project','project');

print(ftpResponseMessage);

 

# Open the file in binary mode

fileObject = open("labelmap.pbtxt", "rb");

file2BeSavedAs = "label.pbtxt"

 

ftpCommand = "STOR %s"%file2BeSavedAs;

 

# Transfer the file in binary mode

ftpResponseMessage = ftpObject.storbinary(ftpCommand, fp=fileObject);

print(ftpResponseMessage);