# Face-Camparison-api
Applied siamese network for face comparision using deeplearing specialization course 4 on coursera for using in my othre project.
App is develoed using bottle.

About Api :

Till now it is working in your local machine only, It can be work on your server but you have to change some configuration for it.

Use it like normal api, 
http://localhost:portnumber/face_comp?img=image_name&name=name

For comaring the faces you should save atleast one image of a person in images folder
and for comapring face you also have to put test image in iamge folder and in url give the name of the test file only
and in name parameter you should give the name of previous image name without extension

It returns only String success if succeed.
Thats all.
