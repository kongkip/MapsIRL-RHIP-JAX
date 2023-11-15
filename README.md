# RHIP Route Optimization

## Project Overview

This project implements the Receding Horizon Inverse Planning (RHIP) algorithm as described in the paper "Massively
Scalable Inverse Reinforcement Learning in Google Maps." The RHIP algorithm is used for route optimization, taking into
account various factors such as distance, traffic, and road conditions, to recommend routes that align with human
preferences.


### Citation
- The paper can be found at [https://ar5iv.org/abs/2305.11290](https://ar5iv.org/abs/2305.11290).


## Features

- Implementation of the RHIP algorithm using Python and Jax.
- Example simulation of route planning on a simplified map.
- Calculation of optimal routes based on inverse reinforcement learning.

## Requirements

- Python 3.x
- Jax (for efficient computation and automatic differentiation)
- Numpy

## Installation

To set up the project, clone the repository and install the required packages.

```
git clone [Your Repository URL]
cd [Your Repository Name]
pip install -r requirements.txt
```

## Usage

To run the RHIP algorithm:

1. Navigate to the project directory.
2. Run the main script:

```
python main.py
```

## Project Structure

- `main.py`: The main script that sets up the environment and runs the RHIP algorithm.
- `rhip.py`: Contains the implementation of the RHIP algorithm and related functions.
- `utils/`: Utility functions and additional modules (if any).

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

## License


## Acknowledgments

Special thanks to OpenAI's GPT-4 for providing guidance and support in implementing the RHIP algorithm and various components of this project.

