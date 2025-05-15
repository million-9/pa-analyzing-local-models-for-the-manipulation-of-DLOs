import numpy as np

from scene_description_p import createScene
import matplotlib.pyplot as plt

def sofa_main():

    # Ignore error messages

    import SofaRuntime
    import Sofa.Gui
    root = Sofa.Core.Node("root")

    # Specify position of contacts as [[contact1-x, contact1-y], [contact2-x, contact2-y], ...]

    # 60,40 for rope_a1, intitial waypoint np.asarray([[30, 20, 30, 1.25*np.pi]])
    # 35,65 for rope, intitial waypoint np.asarray([[30, 40, 30, 1*np.pi]])
    #contact_pos = np.asarray([[35, 65  ]]) #60,40 ,

    # Define sequence of waypoints [[x1, y1, z1, yaw1], [x2, y2, z2, yaw2], ...] that gripper with grasped end should drive([[30, 20, 30, 1.25*np.pi]])
    dp = np.asarray([np.load("rope_ex2.npy")[:48]])
    waypoints = np.asarray([[20, 20, 30, 1.25*np.pi]])
    print(dp.shape)

    # Create SOFA scene
    a=createScene(root,waypoints=waypoints,desired_position=dp,threshold=1,max_data=2000,train_interval=1000)

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

    # Save the array to a .npy file
    rc = np.load('controlpath.npy')
    print(rc.shape)

    FIG, ax = plt.subplots()

    for i in range(0, rc.shape[0], 5):
        ax.cla()  # Clear the axes at the START of each frame, not the end
        ax.set_xlim([-10, 105])
        ax.set_ylim([0, 60])
        ax.grid()

        # Plot contact positions


        # Plot the rope at this frame
        rope_points = rc[i, :48, :]
        ax.plot(rope_points[:, 0], rope_points[:, 1], 'g', label='Rope')
        ax.plot(rope_points[0, 0], rope_points[0, 1], 'r.', label='Rope Start')
        ax.plot(rope_points[-1, 0], rope_points[-1, 1], 'r.', label='Rope End')
        rope_points = dp[-1, :48, :]
        ax.plot(rope_points[:, 0], rope_points[:, 1], 'r', label='Rope')
        ax.plot(rope_points[0, 0], rope_points[0, 1], 'g.', label='Rope Start')
        ax.plot(rope_points[-1, 0], rope_points[-1, 1], 'g.', label='Rope End')

        plt.xlabel("X")
        plt.ylabel("Y")

        # To avoid repeated legend entries, only add the legend on the first frame
        if i == 0:
            plt.legend()

        # Refresh display
        plt.draw()
        plt.pause(0.0001)

    # Keep the final frame visible after the lo
    plt.show()
sofa_main()

