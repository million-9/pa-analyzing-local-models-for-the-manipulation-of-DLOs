import numpy as np
import matplotlib.pyplot as plt
from scene_description import createScene
import Sofa
import Sofa.Core


def sofa_main():

    # Ignore error messages
    import SofaRuntime
    import Sofa.Gui
    root = Sofa.Core.Node("root")

    # Specify position of contacts as [[contact1-x, contact1-y], [contact2-x, contact2-y], ...]
    contact_pos = np.asarray([[60, 60]])

    # Define sequence of waypoints [[x1, y1, z1, yaw1], [x2, y2, z2, yaw2], ...] that gripper with grasped end should drive
    waypoints = np.asarray([[55, 10, 30, 1.75*np.pi]])

    # Create SOFA scene
    a = createScene(root, waypoints=waypoints, contact_pos=contact_pos)

    # Initialize simulation
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    # Initialization of the scene will be done here
    print("Before Main Loop")
    Sofa.Gui.GUIManager.MainLoop(root)
    print("After Main Loop")
    Sofa.Gui.GUIManager.closeGUI()
    print("GUI was closed")

    np.save('rope_ex2.npy',a)


if __name__ == '__main__':
    sofa_main()
