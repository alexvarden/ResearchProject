import pandas as pd
import numpy as np


from sklearn.metrics.pairwise import rbf_kernel



def perturb_with_radial_kernel(x, kernel_width, scale=1):
    """Perturb the given example using a radial kernel function.
    
    Args:
        x (numpy.ndarray or pandas.DataFrame or pandas.Series): The example to perturb.
        kernel_width (float): The width of the radial kernel.
        
    Returns:
        numpy.ndarray: The perturbed example.
    """
    # Convert to NumPy array if necessary
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.to_numpy()

    # Generate random noise
    noise = np.random.normal(0, scale, size=x.shape)


    # # Compute radial kernel
    distances = np.linalg.norm(x - x.reshape(-1, 1), axis=0)

    print(distances)
    kernel = np.exp(-((distances ** 2) / (2 * (kernel_width ** 2))))
    print(kernel)
    # Compute perturbed example
    perturbed_x = x + (noise * kernel)

    return perturbed_x


# def perturb_with_radial_kernel(x, kernel_width, scale=1):
#     """Perturb the given example using a radial kernel function.
    
#     Args:
#         x (numpy.ndarray or pandas.DataFrame or pandas.Series): The example to perturb.
#         kernel_width (float): The width of the radial kernel.
        
#     Returns:
#         numpy.ndarray: The perturbed example.
#     """
#     # Convert to NumPy array if necessary
#     if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
#         x = x.to_numpy()

#     # Generate random noise
#     noise = np.random.normal(0, scale, size=x.shape)

#     print(x)

#     # # Compute radial kernel
#     distances = np.linalg.norm(x - x.reshape(-1, 1), axis=0)

#     print(distances)
#     kernel = np.exp(-((distances ** 2) / (2 * (kernel_width ** 2))))
#     print(kernel)
#     # Compute perturbed example
#     perturbed_x = x + (noise * kernel)

#     return perturbed_x



def generate_perturbations(example, num_perturbations, kernel_width, scale=1):
    """Generate multiple perturbed examples from the given example using a radial kernel function.
    
    Args:
        example (pandas.DataFrame): The example to perturb.
        num_perturbations (int): The number of perturbed examples to generate.
        kernel_width (float): The width of the radial kernel.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the perturbed examples.
    """
     
    category_features = example.select_dtypes(include=['category'])

    
    # One-hot encode categorical features
    example = pd.get_dummies(example)

    # print(example)

    # Generate perturbed examples
    perturbed_examples = []
    for i in range(num_perturbations):
        perturbed_x = perturb_with_radial_kernel(
            example.values, kernel_width, scale)

        # Convert perturbed array to DataFrame
        perturbed_df = pd.DataFrame(perturbed_x, columns=example.columns)

        # Reverse one-hot encoding
        for columName in category_features:
            hotstuff = [col for col in perturbed_df if col.startswith(columName+"_ ")]
            
            # print(perturbed_df[hotstuff])
            
            # perturbed_df[columName] = perturbed_df[hotstuff].idxmax(
            #     axis=1)
            # perturbed_df[columName] = perturbed_df[columName].str.removeprefix(columName+"_ ")
            # perturbed_df = perturbed_df.drop(hotstuff, axis=1)

        perturbed_examples.append(perturbed_df)

    return pd.concat(perturbed_examples)
