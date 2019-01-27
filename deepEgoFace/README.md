



each line of an embedding dataset file should be as follow:
stream-name,frame-number,face-number-in-frame,embedding
e.g.:
vid1,1,1,0.9 0.2 0.5 .....

each line of an embedding and class dataset file should be as follow:
stream-name,frame-number,face-number-in-frame,embedding,class-number
e.g.:
vid1,1,1,0.9 0.2 0.5 .....,1





Extracted face crop of the corpus should be divided in folders of their respective class name
e.g.:
corpus
--class1
----vid1-1-1.jpg
----vid1-2-1.jpg
...
--class2
----vid1-1-2.jpg
----vid2-3-2.jpg
...
...
