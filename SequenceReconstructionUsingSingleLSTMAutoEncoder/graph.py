from matplotlib import pyplot as plt
import numpy as np

def plot_graph(data_input, data_output):
    hip_angles_gt = data_input[:,:,:].permute(0, 1, 2)[:, -1, -1]
    hip_gt=hip_angles_gt.reshape([120,6])
    hip_g=hip_gt[:,1:4]
        #print('gt', hip_gt.shape)
          
    hip_angles_pred = data_output[:,:,:].permute(0, 1, 2)[:, -1, -1]
        #hip_angles_pred=hip_angles_pred.squeeze(1).detach().cpu().numpy()
    hip_pred=hip_angles_pred.reshape([120,6])
    hip_pred=hip_pred.squeeze(1).detach().cpu().numpy()
    hip_p=hip_pred[:,1:4]
        #print('pred', hip_pred.shape)

    time = np.arange(0, len(hip_gt), 1)
            
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.suptitle('Hip right angle')
    ax.plot(time, hip_g[:,1])
    ax.plot(time,hip_p[:,1])
        
            
    fig.tight_layout()
    plt.show()

