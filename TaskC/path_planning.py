import numpy as np


from scene_description_p import createScene
import matplotlib.pyplot as plt

def  sofa_main():

    # Ignore error messages

    import SofaRuntime
    import Sofa.Gui
    root = Sofa.Core.Node("root")

    # Specify position of contacts as [[contact1-x, contact1-y], [contact2-x, contact2-y], ...]

    # 60,40 for rope_a1, intitial waypoint np.asarray([[30, 20, 30, 1.25*np.pi]])
    # 35,65 for rope, intitial waypoint np.asarray([[30, 40, 30, 1*np.pi]])
    contact_pos = np.asarray([[90, 50],[80, 80]])

    # Define sequence of waypoints [[x1, y1, z1, yaw1], [x2, y2, z2, yaw2], ...] that gripper with grasped end should drive
    waypoints = np.asarray([[70, 65, 30, 1 * np.pi], [75, 60, 30, 1.25 * np.pi]])
    desired_poses=np.load("2_contact2.npy")[:9]

    # Create SOFA scene:
    a=createScene(root,waypoints=waypoints,desired_position=desired_poses,threshold=5,contact_pos=contact_pos,max_data=5000,train_interval=1000)

    # Initialize simulation
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1980 , 1080)

    # Initialization of the scene will be done here
    print("Before Main Loop")
    Sofa.Gui.GUIManager.MainLoop(root)
    print("After Main Loop")
    Sofa.Gui.GUIManager.closeGUI()
    print("GUI was closed")

    # Save the array to a .npy file
    rc = np.load('controlpath.npy')
    tp = np.load('time_history.npy')
    print(tp)
    rc = np.load('controlpath.npy')
    fig, ax = plt.subplots()

    for i in range(0, rc.shape[0], 5):
        ax.cla()  # Clear the axes at the START of each frame, not the end
        ax.set_xlim([-20, 185])
        ax.set_ylim([0, 160])
        ax.grid()

        # Plot contact positions
        for c in contact_pos:
            ax.scatter(c[0], c[1], color='grey', s=200, label='Contact')

        # Plot the rope at this frame
        rope_points = rc[i, :85, :]
        ax.plot(rope_points[:, 0], rope_points[:, 1], 'g', label='Rope')
        ax.plot(rope_points[0, 0], rope_points[0, 1], 'r.', label='Rope Start')
        ax.plot(rope_points[-1, 0], rope_points[-1, 1], 'r.', label='Rope End')
        rope_points = desired_poses[-1, :85, :]
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

