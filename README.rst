Face recognition
================
Face_recognition is a face detection and recognition library
which allows you to detect
and recognize faces without using OpenCV's face detection. It's provided a simple face detection and face recognition command line tool that lets you detect and recognize images from a folder using the command line. It was builded using the Dlib face detection models.

Features
________
Finds all the faces that appear in a picture

Here goes the image example.

::

    import face_recognition
    image = face_recognition.load_image_file("imageFile.png")
    face_location = face_recognition.face_location(image)

Face recognition. It can recognize known faces ant tell who they are.

::

    import face_recognition
    image = face_recognition.load_image_file("imageFile.png")
    unknown_image = face_recognition.load_image_file("unknown_image.jpg")

    image_encoding = face_recognition.face_encodings(image)[0]
    unknown_encoding = face_recognition.face_encodings("unknown_image")[0]

    result = face_recognition.compare_faces([image_encoding], unknown_encoding)

Real time face detection

::

    hello here goes the code


Installation
____________

It has been develope in python 3.9 so i don't know if it works in other python versions.

Linux supported(I haven't try it on macOS or windows)

Currently working on...

Reference
_________
I build the library base on ageitey project.
https://github.com/ageitgey/face_recognition
