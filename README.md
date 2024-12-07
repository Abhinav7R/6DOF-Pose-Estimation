# MR_project
Mobile Robotics Course Project

6DoF pose estimation of a 3D object using a single RGB image.
This is done using differentiable rendering method. The 3D object is rendered using pytorch3d and the rendered image is compared with the input image to get the loss. The loss is minimized using backpropagation to get the optimal pose of the object.

1. Installations can be done using installation.ipynb file. It will install pytorch3d which is required to render the 3D models.

2. setup.py is used to start the 6dof pose estimation. It contains functions to load the object(mesh) and the renderers.

3. We have implemented three approaches to optimise the pose:   
    a. Using Rotation matrix and translation vector   
    b. Using Axis angles and translation vector   
    c. Using Quaternion and translation vector   

4. Each approach has 3 files:
    a. python notebook file to run the code and visualise the results
    b. python file which has the training class
    c. python file which has the code to analyse the results

5. The results are saved in the results folder. It contains gifs of the training process to visualise the estimated pose of the object.

6. The data folder contains the 3D meshes of the objects and initialisation of the pose(translation and rotation) of the object to keep it uniform across all the approaches.


Contributors:
1. Meet Gera
2. Abhinav Raundhal
3. Amey Karan