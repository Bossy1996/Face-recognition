Face recognition
================
Face_recognition is a face detection and recognition library
which allows you to detect
and recognize faces without using OpenCV.

Built using Dlib models.

It's provided a simple face detection and face recognition command line tool that lets you detect and recognize images from a folder using the command line.

Features
________
Finds all the faces that appear in a picture

Here goes the image example.

::

    import face_recognition
    image = face_recognition.load_image_file("imageFile.png")
    face_location = face_recognition.face_location(image)

Face recognition