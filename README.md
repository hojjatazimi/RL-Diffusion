# RL-Diffusion

RL-Diffusion is a reinforcement learning (RL) framework that integrates diffusion models to solve complex tasks through probabilistic modeling and policy optimization. The project is designed to explore the intersection of RL and generative models, particularly focusing on probabilistic state transitions and action selection.

Table of Contents
	•	Overview
	•	Features
	•	Installation
	•	Usage
	•	Training
	•	Contributing
	•	License

Overview

RL-Diffusion uses a combination of diffusion models and reinforcement learning techniques to train policies in complex environments. The framework is built with PyTorch and provides a modular structure for integrating custom policy networks and reward mechanisms.

Key Highlights:
	•	Probabilistic action selection using diffusion models.
	•	Fully customizable policy networks and reward functions.
	•	Built-in tools for gradient monitoring and NaN handling during training.

Features
	•	Diffusion-based RL: Combines forward and reverse processes of diffusion models with RL for better action-state modeling.
	•	Policy Network Integration: Easily define and train your custom policy networks.
	•	Error Handling: Automatic detection of NaN values in gradients, losses, and parameters during training.
	•	Modular Design: Extendable and adaptable framework to suit diverse use cases.

Installation
	1.	Clone the repository:

git clone https://github.com/yourusername/RL-Diffusion.git
cd RL-Diffusion


	2.	Install dependencies:

pip install -r requirements.txt


	3.	Ensure your environment supports PyTorch:

# Install PyTorch (adjust for your system and CUDA version)
pip install torch torchvision torchaudio

Usage

Defining Your Model

Modify the PolicyNetwork and any custom reward functions in the /src/classes directory as per your requirements.

Training

The main training loop is implemented in train_rl located in src/train.ipynb. You can adjust the following parameters:
	•	Epochs (nb_epochs)
	•	Batch Size (batch_size)
	•	Learning Rate (defined in the optimizer)

Example training command:

from src.train import train_rl
training_loss, rewards = train_rl(model=policy_net, optimizer=optimizer)

Training

During training, the model will:
	1.	Generate state-action pairs using diffusion processes.
	2.	Calculate rewards based on the defined reward function.
	3.	Optimize the policy network to maximize the cumulative reward.

Monitor training logs to ensure gradients, parameters, and rewards are stable. Any NaN detections are flagged during training.

Contributing

Contributions are welcome! To contribute:
	1.	Fork the repository.
	2.	Create a feature branch:

git checkout -b feature-name


	3.	Commit your changes:

git commit -m "Add new feature"


	4.	Push to your branch:

git push origin feature-name


	5.	Submit a pull request.

License

This project is licensed under the MIT License. Feel free to use and modify it for your needs.

Acknowledgments

Special thanks to the contributors and the open-source community for their invaluable tools and resources.