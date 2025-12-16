import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self, num_links, num_channels):
        self.num_links = num_links
        self.num_channels = num_channels
        
        # Enable interactive mode
        plt.ion()
        
        # --- Figure 1: Reward and Loss ---
        self.fig1, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        
        self.line_reward, = self.ax1.plot([], [], 'b-', label='Reward')
        self.line_loss, = self.ax2.plot([], [], 'r-', label='Loss')
        
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Reward', color='b')
        self.ax2.set_ylabel('Loss', color='r')
        
        self.fig1.suptitle('Training Progress')
        self.fig1.legend([self.line_reward, self.line_loss], ['Reward', 'Loss'])
        self.fig1.tight_layout()
        
        # --- Figure 2: Channel Probabilities ---
        self.fig2, self.axes2 = plt.subplots(num_links, 1, figsize=(8, 2*num_links), sharex=True)
        if num_links == 1:
            self.axes2 = [self.axes2]
            
        self.lines_probs = []
        self.lines_off = []
        
        for i in range(num_links):
            ap_lines = []
            ax = self.axes2[i]
            
            # Lines for each channel
            for c in range(num_channels):
                line, = ax.plot([], [], label=f'Ch {c+1}')
                ap_lines.append(line)
            self.lines_probs.append(ap_lines)
            
            # Line for Off probability
            line_off, = ax.plot([], [], 'k--', label='Off')
            self.lines_off.append(line_off)
            
            ax.set_ylabel(f'AP {i} Prob')
            ax.set_ylim(0, 1.05) # Slightly above 1 to see the line clearly
            if i == 0:
                ax.legend(loc='upper right', fontsize='small', ncol=2)
                
        self.axes2[-1].set_xlabel('Epoch')
        self.fig2.suptitle('Avg Channel & Off Probabilities per Epoch')
        self.fig2.tight_layout()
        
        # Data storage
        self.rewards = []
        self.losses = []
        # history_probs[ap_idx][channel_idx] -> list
        self.history_probs = [[[] for _ in range(num_channels)] for _ in range(num_links)]
        # history_off[ap_idx] -> list
        self.history_off = [[] for _ in range(num_links)]

    def update(self, reward, loss, avg_channel_probs, avg_off_probs):
        """
        Update the plots with new data from the latest epoch.
        
        :param reward: float, average reward of the epoch
        :param loss: float, average loss of the epoch
        :param avg_channel_probs: np.array of shape (num_links, num_channels)
        :param avg_off_probs: np.array of shape (num_links,)
        """
        # --- Update Reward/Loss ---
        self.rewards.append(reward)
        self.losses.append(loss)
        
        epochs_range = range(len(self.rewards))
        
        self.line_reward.set_xdata(epochs_range)
        self.line_reward.set_ydata(self.rewards)
        self.line_loss.set_xdata(epochs_range)
        self.line_loss.set_ydata(self.losses)
        
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # --- Update Channel/Off Probs ---
        for i in range(self.num_links):
            # Channels
            for c in range(self.num_channels):
                val = avg_channel_probs[i, c]
                self.history_probs[i][c].append(val)
                self.lines_probs[i][c].set_xdata(range(len(self.history_probs[i][c])))
                self.lines_probs[i][c].set_ydata(self.history_probs[i][c])
            
            # Off
            val_off = avg_off_probs[i]
            self.history_off[i].append(val_off)
            self.lines_off[i].set_xdata(range(len(self.history_off[i])))
            self.lines_off[i].set_ydata(self.history_off[i])
            
            self.axes2[i].relim()
            self.axes2[i].autoscale_view()
            
        # Draw
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        
        # Pause to allow GUI update
        plt.pause(0.01)

    def save_plots(self, prefix='training_results'):
        """
        Save the current figures as PDF files.
        """
        self.fig1.savefig(f'{prefix}_reward_loss.pdf')
        self.fig2.savefig(f'{prefix}_probs.pdf')
        print(f"Saved plots to {prefix}_reward_loss.pdf and {prefix}_probs.pdf")

    def close(self):
        plt.ioff()
        # plt.show() # Optional
