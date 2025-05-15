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
    contact_pos = np.asarray([[90, 50],[80, 80]])

    #[[75+20, 70, 30, 1*np.pi],,[90+20, 100, 30, 1*np.pi],[30+20, 100, 30, 1*np.pi],[50+20, 60, 30, 1.5*np.pi],[100, 70, 30, 0*np.pi],[110, 30, 30, 1.5*np.pi],[75, 40, 30, 1*np.pi]]

    # Define sequence of waypoints [[x1, y1, z1, yaw1], [x2, y2, z2, yaw2], ...] that gripper with grasped end should drive
    waypoints = np.asarray([[80, 70, 30, 1*np.pi],[90, 70, 30, 1.3*np.pi],
                            [110, 100, 30, 1.5*np.pi],[100, 100, 30, 1.3*np.pi],
                            [55, 80, 30, 1.5 * np.pi],
                            [70, 65, 30, 1.75 * np.pi],
                            [90, 65, 30, 0 * np.pi],
                            [110, 60, 30, 1.75* np.pi],
                            [100, 40, 30, 1.5* np.pi],
                            [90, 40, 30, 1.25* np.pi],
                            [80, 40, 30, 1 * np.pi],



                             ])

    # Create SOFA scene
    a = createScene(root, waypoints=waypoints, contact_pos=contact_pos)

    # Initialize simulation
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1980, 1080)
    # Initialization of the scene will be done here
    print("Before Main Loop")
    Sofa.Gui.GUIManager.MainLoop(root)
    print("After Main Loop")
    Sofa.Gui.GUIManager.closeGUI()
    print("GUI was closed")

    np.save('dc2_c9.npy',a)


if __name__ == '__main__':
    sofa_main()
