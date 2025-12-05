This EKF will be a Map-Based VIO (Localization) filter for now(We need to delete this file after updating the ekf). We assume we know the location of the Red and Green cylinders from our world file, and we use them to correct the robot's drifting IMU position.
The Math (Simplified)

    State: x=[px​,py​,pz​,vx​,vy​,vz​,qw​,qx​,qy​,qz​,bax​,bay​,baz​,bgx​,bgy​,bgz​] (16 elements).

    Prediction: Integrate IMU accel/gyro to update p,v,q. Increase covariance P.

    Update: When we see a red cylinder:

        Project known Red Cylinder position into camera frame.

        Compare with observed (u,v) pixels.

        Calculate Kalman Gain K and correct x.