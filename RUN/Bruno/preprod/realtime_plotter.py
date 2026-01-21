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
        # Legend outside
        self.fig1.legend([self.line_reward, self.line_loss], ['Reward', 'Loss'], 
                         loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        self.fig1.tight_layout()
        # Adjust layout to make room for legend
        self.fig1.subplots_adjust(right=0.85)
        
        # Annotations for Figure 1
        self.text_last_reward = self.ax1.text(0, 0, '', color='b', fontsize=8)
        self.text_last_loss = self.ax2.text(0, 0, '', color='r', fontsize=8)
        
        self.text_max_reward = self.ax1.text(0, 0, '', color='b', fontsize=8, fontweight='bold')
        self.text_min_loss = self.ax2.text(0, 0, '', color='r', fontsize=8, fontweight='bold')
        
        # Markers for max/min
        self.marker_max_reward, = self.ax1.plot([], [], 'bo')
        self.marker_min_loss, = self.ax2.plot([], [], 'ro')
        
        # --- Figure 2: Channel Probabilities ---
        self.fig2, self.axes2 = plt.subplots(num_links, 1, figsize=(8, 2*num_links), sharex=True)
        if num_links == 1:
            self.axes2 = [self.axes2]
            
        self.lines_probs = []
        self.lines_off = []
        self.texts_probs = []
        self.texts_off = []
        
        for i in range(num_links):
            ap_lines = []
            ap_texts = []
            ax = self.axes2[i]
            
            # Lines for each channel
            for c in range(num_channels):
                line, = ax.plot([], [], label=f'Ch {c+1}')
                ap_lines.append(line)
                # Text for last value
                text = ax.text(0, 0, '', fontsize=8, color=line.get_color())
                ap_texts.append(text)
            self.lines_probs.append(ap_lines)
            self.texts_probs.append(ap_texts)
            
            # Line for Off probability
            line_off, = ax.plot([], [], 'k--', label='Off')
            self.lines_off.append(line_off)
            # Text for Off value
            text_off = ax.text(0, 0, '', fontsize=8, color='k')
            self.texts_off.append(text_off)
            
            ax.set_ylabel(f'AP {i} Prob')
            ax.set_ylim(0, 1.05) # Slightly above 1 to see the line clearly
            ax.set_ylabel(f'AP {i} Prob')
            ax.set_ylim(0, 1.05) # Slightly above 1 to see the line clearly
            if i == 0:
                # Legend outside
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', ncol=1)
                
        self.axes2[-1].set_xlabel('Epoch')
        self.axes2[-1].set_xlabel('Epoch')
        self.fig2.suptitle('Avg Channel & Off Probabilities per Epoch')
        self.fig2.tight_layout()
        self.fig2.subplots_adjust(right=0.85)
        
        # --- Figure 3: Mu Evolution ---
        self.fig3, self.axes3 = plt.subplots(num_links, 1, figsize=(8, 2*num_links), sharex=True)
        if num_links == 1:
            self.axes3 = [self.axes3]
            
        self.lines_mu = []
        self.texts_mu = []
        
        for i in range(num_links):
            ax = self.axes3[i]
            line, = ax.plot([], [], 'g-', label=f'Mu AP {i}')
            self.lines_mu.append(line)
            
            text = ax.text(0, 0, '', fontsize=8, color='g')
            self.texts_mu.append(text)
            
            ax.set_ylabel(f'Mu AP {i}')
            if i == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
                
        self.axes3[-1].set_xlabel('Epoch')
        self.fig3.suptitle('Mu Evolution per AP')
        self.fig3.tight_layout()
        self.fig3.subplots_adjust(right=0.85)

        # Data storage
        self.rewards = []
        self.losses = []
        # history_probs[ap_idx][channel_idx] -> list
        self.history_probs = [[[] for _ in range(num_channels)] for _ in range(num_links)]
        # history_off[ap_idx] -> list
        self.history_off = [[] for _ in range(num_links)]
        # history_mu[ap_idx] -> list
        self.history_mu = [[] for _ in range(num_links)]

    def update(self, reward, loss, avg_channel_probs, avg_off_probs, avg_mu):
        """
        Update the plots with new data from the latest epoch.
        
        :param reward: float, average reward of the epoch
        :param loss: float, average loss of the epoch
        :param avg_channel_probs: np.array of shape (num_links, num_channels)
        :param avg_off_probs: np.array of shape (num_links,)
        :param avg_mu: np.array of shape (num_links,)
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
        
        # --- Annotations Figure 1 ---
        # Last values
        self.text_last_reward.set_text(f'{reward:.2f}')
        self.text_last_reward.set_position((len(self.rewards)-1, reward))
        
        self.text_last_loss.set_text(f'{loss:.2f}')
        self.text_last_loss.set_position((len(self.losses)-1, loss))
        
        # Max Reward
        max_r = np.max(self.rewards)
        max_r_idx = np.argmax(self.rewards)
        self.text_max_reward.set_text(f'Max: {max_r:.2f}')
        self.text_max_reward.set_position((max_r_idx, max_r))
        self.marker_max_reward.set_data([max_r_idx], [max_r])
        
        # Min Loss
        min_l = np.min(self.losses)
        min_l_idx = np.argmin(self.losses)
        self.text_min_loss.set_text(f'Min: {min_l:.2f}')
        self.text_min_loss.set_position((min_l_idx, min_l))
        self.marker_min_loss.set_data([min_l_idx], [min_l])
        
        # --- Resolve Overlaps for Figure 1 (Last values) ---
        # We need to handle dual axes (different scales)
        # 1. Get limits
        r_min, r_max = self.ax1.get_ylim()
        l_min, l_max = self.ax2.get_ylim()
        
        # 2. Normalize values to [0, 1]
        r_range = r_max - r_min if r_max != r_min else 1.0
        l_range = l_max - l_min if l_max != l_min else 1.0
        
        r_norm = (reward - r_min) / r_range
        l_norm = (loss - l_min) / l_range
        
        # 3. Check overlap in normalized space
        # We want at least 5% separation
        min_dist_norm = 0.05
        
        # Create items with metadata
        # Initialize with Last Reward and Last Loss
        items_fig1 = [
            {'val': r_norm, 'type': 'reward', 'range': r_range, 'min': r_min, 'text': self.text_last_reward, 'x': len(self.rewards)-1},
            {'val': l_norm, 'type': 'loss', 'range': l_range, 'min': l_min, 'text': self.text_last_loss, 'x': len(self.losses)-1}
        ]
        
        # Check Max Reward and Min Loss for X-proximity
        current_x = len(self.rewards) - 1
        x_thresh = max(5, current_x * 0.15) # 15% of plot width
        
        # Max Reward
        max_r_idx = np.argmax(self.rewards)
        if abs(max_r_idx - current_x) < x_thresh:
             max_r = np.max(self.rewards)
             max_r_norm = (max_r - r_min) / r_range
             items_fig1.append({'val': max_r_norm, 'type': 'reward', 'range': r_range, 'min': r_min, 'text': self.text_max_reward, 'x': max_r_idx})

        # Min Loss
        min_l_idx = np.argmin(self.losses)
        if abs(min_l_idx - current_x) < x_thresh:
             min_l = np.min(self.losses)
             min_l_norm = (min_l - l_min) / l_range
             items_fig1.append({'val': min_l_norm, 'type': 'loss', 'range': l_range, 'min': l_min, 'text': self.text_min_loss, 'x': min_l_idx})
        
        # Sort by normalized height
        items_fig1.sort(key=lambda x: x['val'])
        
        # Push apart if needed (Iterative)
        for i in range(1, len(items_fig1)):
            prev = items_fig1[i-1]
            curr = items_fig1[i]
            if (curr['val'] - prev['val']) < min_dist_norm:
                # Push current up
                curr['val'] = prev['val'] + min_dist_norm
            
        # 4. Apply back to data coordinates
        for item in items_fig1:
            new_val_data = item['val'] * item['range'] + item['min']
            # Update position
            item['text'].set_position((item['x'], new_val_data))
        
        # --- Update Channel/Off Probs ---
        for i in range(self.num_links):
            # Channels
            for c in range(self.num_channels):
                val = avg_channel_probs[i, c]
                self.history_probs[i][c].append(val)
                x_data = range(len(self.history_probs[i][c]))
                self.lines_probs[i][c].set_xdata(x_data)
                self.lines_probs[i][c].set_ydata(self.history_probs[i][c])
                
                # Update text annotation
                self.texts_probs[i][c].set_text(f'{val:.2f}')
                self.texts_probs[i][c].set_position((len(self.history_probs[i][c]) - 1, val))
            
            # Off
            val_off = avg_off_probs[i]
            self.history_off[i].append(val_off)
            x_data_off = range(len(self.history_off[i]))
            self.lines_off[i].set_xdata(x_data_off)
            self.lines_off[i].set_ydata(self.history_off[i])
            
            # Update text annotation (initial position, will be adjusted)
            self.texts_off[i].set_text(f'{val_off:.2f}')
            self.texts_off[i].set_position((len(self.history_off[i]) - 1, val_off))
            
            # --- Resolve Overlaps for this AP ---
            # Collect all texts and their y-values
            texts_to_adjust = []
            # Channels
            for c in range(self.num_channels):
                texts_to_adjust.append({'val': avg_channel_probs[i, c], 'text': self.texts_probs[i][c]})
            # Off
            texts_to_adjust.append({'val': avg_off_probs[i], 'text': self.texts_off[i]})
            
            self._resolve_overlaps(texts_to_adjust)

            self.axes2[i].relim()
            self.axes2[i].autoscale_view()

        # --- Update Mu ---
        for i in range(self.num_links):
            val_mu = avg_mu[i]
            self.history_mu[i].append(val_mu)
            x_data_mu = range(len(self.history_mu[i]))
            self.lines_mu[i].set_xdata(x_data_mu)
            self.lines_mu[i].set_ydata(self.history_mu[i])
            
            self.texts_mu[i].set_text(f'{val_mu:.2f}')
            self.texts_mu[i].set_position((len(self.history_mu[i]) - 1, val_mu))
            
            self.axes3[i].relim()
            self.axes3[i].autoscale_view()
            
        # Draw
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        self.fig3.canvas.draw()
        self.fig3.canvas.flush_events()
        
        # Pause to allow GUI update
        plt.pause(0.01)

    def _resolve_overlaps(self, items, min_dist=0.08):
        """
        Adjust vertical positions of text labels to avoid overlap.
        items: list of dicts {'val': float, 'text': Text}
        """
        # Sort by value (y-coordinate)
        items.sort(key=lambda x: x['val'])
        
        if not items:
            return

        # Simple iterative adjustment
        # We will spread them out if they are too close
        
        # 1. Assign ideal positions (actual values)
        for item in items:
            item['y_pos'] = item['val']
            
        # 2. Push apart
        # We do multiple passes or a single pass from bottom up?
        # Let's try a simple bottom-up push
        for i in range(1, len(items)):
            prev = items[i-1]
            curr = items[i]
            
            dist = curr['y_pos'] - prev['y_pos']
            if dist < min_dist:
                # Move current up
                curr['y_pos'] = prev['y_pos'] + min_dist
        
        # 3. Apply positions
        # We need the x-coordinate (last epoch)
        # Assuming all texts are at the same x (current epoch)
        # We can get x from the text object itself or pass it.
        # The text object already has the x set from the loop above.
        
        for item in items:
            x, _ = item['text'].get_position()
            item['text'].set_position((x, item['y_pos']))

    def save_plots(self, prefix='training_results'):
        """
        Save the current figures as PDF files.
        """
        self.fig1.savefig(f'{prefix}_reward_loss.pdf')
        self.fig2.savefig(f'{prefix}_probs.pdf')
        self.fig3.savefig(f'{prefix}_mu.pdf')
        print(f"Saved plots to {prefix}_reward_loss.pdf, {prefix}_probs.pdf and {prefix}_mu.pdf")

    def close(self):
        plt.ioff()
        # plt.show() # Optional
