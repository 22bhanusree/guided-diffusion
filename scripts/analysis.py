import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from meansparse_unet import MeanSparse

def main():
   

    dist_util.setup_dist()
    
    # Load the saved classifiers
    def load_classifiers(filename='meansparse_classifiers.pth'):
        classifiers = th.load(filename)
        
        return classifiers

    
    
    # load_classifiers('../models/64x64_classifier.pt')
    loaded_meansparse_classifiers_ddp = load_classifiers('meansparse_classifiers_temp_1.pth')
    loaded_meansparse_classifiers_ddp_2 = load_classifiers('meansparse_classifiers_temp_2.pth')
    loaded_meansparse_classifiers_ddp_3 = load_classifiers('meansparse_classifiers_temp_3.pth')
    loaded_meansparse_classifiers_ddp_4 = load_classifiers('meansparse_classifiers_temp_4.pth')
    loaded_meansparse_classifiers_ddp_5 = load_classifiers('meansparse_classifiers_temp_5.pth')
   
    # state_dict = loaded_meansparse_classifiers_ddp[0].state_dict()
    # for key in state_dict:
    #     print(key)
    # print(f'meansparse classifiers {loaded_meansparse_classifiers_ddp}')

    def extract_running_statistics(classifiers, layer_name):
        '''
        Extract running mean and running variance from classifiers.
        
        :param classifiers: List of classifiers.
        :param layer_name: Name of the layer to extract statistics from.
        :return: Dictionary containing running means and variances.
        '''
        running_means = {}
        running_vars = {}
        
        for idx, classifier in enumerate(classifiers):
            state_dict = classifier.state_dict()
            running_mean_key = f'{layer_name}.running_mean'
            running_var_key = f'{layer_name}.running_var'
            
            if running_mean_key in state_dict and running_var_key in state_dict:
                running_means[idx] = state_dict[running_mean_key].cpu().numpy()
                running_vars[idx] = state_dict[running_var_key].cpu().numpy()
        
        return running_means, running_vars


    # Extract running means and variances
    layer_name = 'module.input_blocks.4.0.meansparse'  # Replace with the actual layer name
    # layer_name = 'module.middle_block.0.meansparse'
    running_means_1, running_vars_1 = extract_running_statistics(loaded_meansparse_classifiers_ddp, layer_name)
    running_means_2, running_vars_2 = extract_running_statistics(loaded_meansparse_classifiers_ddp_2, layer_name)
    running_means_3, running_vars_3 = extract_running_statistics(loaded_meansparse_classifiers_ddp_3, layer_name)
    running_means_4, running_vars_4 = extract_running_statistics(loaded_meansparse_classifiers_ddp_4, layer_name)
    running_means_5, running_vars_5 = extract_running_statistics(loaded_meansparse_classifiers_ddp_5, layer_name)


    def plot_print(running_means_list, running_vars_list, mean_filename='running_means', var_filename='running_vars'):
        """
        Plots, prints, and saves the mean and variance values across dimensions for each classifier set at specific x-axis positions.

        :param running_means_list: List of dictionaries containing running means for each classifier set.
        :param running_vars_list: List of dictionaries containing running variances for each classifier set.
        :param mean_filename: Filename prefix for the saved running means plot.
        :param var_filename: Filename prefix for the saved running variances plot.
        """
        positions = [0, 200, 400, 600, 800]  # Specific x-axis positions for each set

        colors = ['r', 'g', 'b', 'c', 'm']
        labels = [f'Model {i+1} at t ={positions[i]}' for i in range(len(running_means_list))]

        # Plot and save running mean values across dimensions
        plt.figure(figsize=(60, 20))
        for i, running_means in enumerate(running_means_list):
            for key, running_mean in running_means.items():
                dimensions = np.arange(1, len(running_mean) + 1)  # X-axis as dimension indices
                plt.plot(dimensions, running_mean, color=colors[i], label=f'{labels[i]} ', marker='o',alpha=0.5)
            # Print running means for debugging
            print(f"Running Means for {labels[i]}:\n{running_means}\n")

        plt.title('Running Mean Values Across Dimensions')
        plt.xlabel('Dimension Index')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{mean_filename}.png', dpi=300)
        
        plt.close()  # Close the figure to avoid overlap

        # Plot and save running variance values across dimensions
        plt.figure(figsize=(60, 20))
        for i, running_vars in enumerate(running_vars_list):
            for key, running_var in running_vars.items():
                dimensions = np.arange(1, len(running_var) + 1)  # X-axis as dimension indices
                plt.plot(dimensions, running_var, color=colors[i], label=f'{labels[i]} ', marker='o',alpha=0.5)
            # Print running variances for debugging
            print(f"Running Variances for {labels[i]}:\n{running_vars}\n")

        plt.title('Running Variance Values Across Dimensions')
        plt.xlabel('Dimension Index')
        plt.ylabel('Variance Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{var_filename}.png', dpi=300)
        
        plt.close()  # Close the figure to avoid overlap

   
    def scatter_print(running_means_list, running_vars_list, mean_filename='running_means', var_filename='running_vars'):
        """
        Plots, prints, and saves the mean and variance values across dimensions for each classifier set.

        :param running_means_list: List of dictionaries containing running means for each classifier set.
        :param running_vars_list: List of dictionaries containing running variances for each classifier set.
        :param mean_filename: Filename prefix for the saved running means plot.
        :param var_filename: Filename prefix for the saved running variances plot.
        """
        positions = [0, 200, 400, 600, 800]  # Specific x-axis positions for each set

        colors = ['r', 'g', 'b', 'c', 'm']
        labels = [f'Model {i+1} at t ={positions[i]}' for i in range(len(running_means_list))]

        # Plot and save running mean values across dimensions
        plt.figure(figsize=(60, 20))
        for i, running_means in enumerate(running_means_list):
            for key, running_mean in running_means.items():
                dimensions = np.arange(1, len(running_mean) + 1)  # X-axis as dimension indices
                plt.scatter(dimensions, running_mean, color=colors[i], label=f'{labels[i]} ', marker='o')
            # Print running means for debugging
            print(f"Running Means for {labels[i]}:\n{running_means}\n")

        plt.title('Running Mean Values Across Dimensions')
        plt.xlabel('Dimension Index')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{mean_filename}.png', dpi=300)
        
        plt.close()  # Close the figure to avoid overlap

        # Plot and save running variance values across dimensions
        plt.figure(figsize=(60, 20))
        for i, running_vars in enumerate(running_vars_list):
            for key, running_var in running_vars.items():
                dimensions = np.arange(1, len(running_var) + 1)  # X-axis as dimension indices
                plt.scatter(dimensions, running_var, color=colors[i], label=f'{labels[i]} Dimension {key}', marker='o')
            # Print running variances for debugging
            print(f"Running Variances for {labels[i]}:\n{running_vars}\n")

        plt.title('Running Variance Values Across Dimensions')
        plt.xlabel('Dimension Index')
        plt.ylabel('Variance Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{var_filename}.png', dpi=300)
        
        plt.close()  # Close the figure to avoid overlap

    # Example usage
    running_means_list = [running_means_1, running_means_2, running_means_3, running_means_4, running_means_5]
    running_vars_list = [running_vars_1, running_vars_2, running_vars_3, running_vars_4, running_vars_5]

    # Plot, print, and save all means and variances across dimensions
    # plot_print(running_means_list, running_vars_list, mean_filename='new_running_means_mid_0', var_filename='new_running_vars_mid_0')
    # scatter_print(running_means_list, running_vars_list, mean_filename='new1_running_means_mid_0', var_filename='new1_running_vars_mid_0')

    import numpy as np

    def linear_interpolate(t1, t2, t_target, mean1, mean2):
        """
        Perform linear interpolation for running means between two time points.

        Args:
            t1 (int): Time of first point.
            t2 (int): Time of second point.
            t_target (int): Time of the target point for interpolation.
            mean1 (np.array): Running mean at time t1.
            mean2 (np.array): Running mean at time t2.

        Returns:
            np.array: Interpolated running mean at time t_target.

        """
        for key , mean1_value in mean1.items():
            mean1 = np.array(mean1[key])  # Extract the array from the dictionary
        for key , mean2_value in mean2.items():
            mean2 = np.array(mean2[key])
        return mean1 + ((t_target - t1) / (t2 - t1)) * (mean2 - mean1)


    # Time points corresponding to the running means
    t1, t2 = 200 , 800
    t_target = 400
    mean1 = running_means_2
    mean2 = running_means_5
    # Interpolation of running means at t=200 between t=0 and t=400
    interpolated_mean_400 = linear_interpolate(t1, t2, t_target, mean1 , mean2)
    

    def plot_real_vs_interpolated_mean(running_mean_real, running_mean_interpolated, time_point, dimension_label="Dimension Index"):
        """
        Function to plot real mean vs interpolated mean at a specific time point.

        Args:
            running_mean_real (np.array): Real running mean values at the given time point.
            running_mean_interpolated (np.array): Interpolated running mean values at the given time point.
            time_point (int): The specific time point (e.g., t=200) for which the plot is made.
            dimension_label (str): Label for the x-axis representing the dimension indices (default: "Dimension Index").
        """
        # Colors and labels
        colors = ['r', 'b']  # Red for real mean, blue for interpolated mean
        labels = [f'Real Mean at t={time_point}', f'Interpolated Mean at t={time_point}']
        

        for key , mean_value in running_mean_real.items():
            running_mean_real= mean_value
        # X-axis as dimension indices
        dimensions = np.arange(1, len(running_mean_real) + 1)

        # Create a plot
        plt.figure(figsize=(20, 8))

        # Plot real mean at the specified time point
        plt.plot(dimensions, running_mean_real, color=colors[0], label=labels[0], marker='o', alpha=0.5)

        # Plot interpolated mean at the specified time point
        plt.plot(dimensions, running_mean_interpolated, color=colors[1], label=labels[1], marker='x', alpha=0.5)

        # Set title and labels
        plt.title(f'Real Mean vs Interpolated Mean at t={time_point} Across Dimensions')
        plt.xlabel(dimension_label)
        plt.ylabel('Mean Value')

        # Add legend and grid
        plt.legend()
        plt.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save plot if a filename is provided
        save_filename = "real_vs_interpolated_mean_t400_using_t200_t800.png"

        if save_filename:
            plt.savefig(save_filename, dpi=300)
            print(f"Plot saved as {save_filename}")

    # Example usage of the function
    plot_real_vs_interpolated_mean(running_means_3, interpolated_mean_400, 400)


if __name__ == "__main__":
    main()






