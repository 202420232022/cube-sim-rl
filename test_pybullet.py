import pybullet as p
import pybullet_data
import time

def main():
    # 1. Connect to the physics server (GUI mode allows us to see the simulation)
    physicsClient = p.connect(p.GUI)
    
    # 2. Add the path to default pybullet assets (plane, R2D2, etc.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 3. Set gravity (Z axis points up in PyBullet)
    p.setGravity(0, 0, -9.81)
    
    # 4. Load a ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # 5. Load a default cube, dropping it from a height of 2 meters
    startPos = [0, 0, 2]
    startOrientation = p.getQuaternionFromEuler([0.5, 0.5, 0]) # Tilted slightly
    
    # We use a built-in cube URDF for this test
    cubeId = p.loadURDF("cube.urdf", startPos, startOrientation)
    
    print("Environment setup complete. Running simulation...")
    
    # 6. Step the simulation forward in time (240Hz by default)
    for i in range(1000):
        p.stepSimulation()
        time.sleep(1./240.) # Real-time delay so we can watch it fall
        
        # Optional: Print the cube's position every 100 steps
        if i % 100 == 0:
            cubePos, cubeOrn = p.getBasePositionAndOrientation(cubeId)
            print(f"Step {i}: Z-height = {cubePos[2]:.2f}m")
            
    # 7. Disconnect when done
    p.disconnect()
    print("Test finished successfully!")

if __name__ == '__main__':
    main()
