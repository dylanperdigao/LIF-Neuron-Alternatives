import numpy as np
import torch
from snntorch import utils
from torch.nn import Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Sequential, Conv1d, Conv2d, MaxPool1d, MaxPool2d, Flatten, Linear
from modules.rate_coding import ce_count_loss
from snntorch import Alpha, Leaky, Lapicque, Synaptic, RLeaky, RSynaptic
from snntorch.surrogate import fast_sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ModelSpikingNeuron(object):
    def __init__(self, neuron: str, classes: int, dataset: str, **kwargs):
        """Initialize the model.

        Args:
            num_classes (int): Number of classes.
            dataset (str): Dataset name.
            **kwargs: Additional arguments.

        Keyword Args:
            epochs (int, optional): Number of epochs. Defaults to 10.
            num_steps (int, optional): Number of simulation steps. Defaults to 50.
            gpu_number (int, optional): GPU number to use. Defaults to None.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            class_weights (torch.Tensor, optional): Class weights. Defaults to None.
        """
        self.classes = classes
        self.dataset = dataset
        self.neuron = neuron
        self.device = self._get_device(kwargs.get('gpu_number', None))
        self.network = self._get_network(neuron, classes, **kwargs).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=kwargs.get('lr', 1e-3), weight_decay=kwargs.get('weight_decay', 0))
        self.epochs: int = kwargs.get('epochs', 10)
        self.num_steps: int = kwargs.get('num_steps', 50)
        self.class_weights: torch.Tensor = kwargs.get('class_weights', None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
        self.loss_fn = ce_count_loss(self.class_weights)
        self.loss_val: torch.Tensor = None
        self.loss_history = []

    def fit(self, train_loader: DataLoader):
        """Training loop for the network.
        
        Args:
            train_loader (DataLoader): Training data.
        """
        for epoch in range(self.epochs):
            print(f"Epoch - {epoch}")
            for i, (data, labels) in enumerate(iter(train_loader)):
                data: torch.Tensor = data.to(self.device)
                labels: torch.Tensor = labels.to(self.device, dtype=torch.long)
                self.network.train()
                spk_rec, _ = self.forward_pass(data=data)
                self.loss_val: torch.Tensor = self.loss_fn(spk_rec, labels)
                self.optimizer.zero_grad()
                self.loss_val.backward(retain_graph=True if self.neuron in ['rsynaptic','lapicque'] else None)
                self.optimizer.step()
            self.loss_history.append(self.loss_val.item())
            print(f"Loss: {self.loss_val.item()}") 
               


    def predict(self, test_loader: DataLoader):
        """Predict the output of the network.

        Args:
            test_loader (DataLoader): Test data.

        Returns:
            tuple: Predictions and test targets.
        """
        predictions = np.array([])
        targets = np.array([])
        with torch.no_grad():
            self.network.eval()
            for i, (data, labels) in enumerate(iter(test_loader)):
                data: torch.Tensor = data.to(self.device)
                labels: torch.Tensor = labels.to(self.device, dtype=torch.long)
                spk_rec, _ = self.forward_pass(data=data)
                spike_code = self.loss_fn.spike_code(spk_rec)
                predictions = np.append(predictions, spike_code.cpu().numpy())
                targets = np.append(targets, labels.cpu().numpy())
        return predictions, targets
    
    def evaluate(self, targets: np.ndarray, predicted: np.ndarray):
        """Evaluate the model using the confusion matrix and some metrics.

        Args:
            targets (np.ndarray): list of true values
            predicted (np.ndarray): list of predicted values
  
        Returns:
            dict: Evaluation metrics
        """
        if self.classes == 2:
            cm = confusion_matrix(targets, predicted)
            tn, fp, _, _ = cm.ravel()
            accuracy = accuracy_score(targets, predicted)
            precision = precision_score(targets, predicted)
            recall = recall_score(targets, predicted)
            f1 = f1_score(targets, predicted)
            fpr = fp / (fp + tn)
            auc = roc_auc_score(targets, predicted)
        else:
            try:
                accuracy = accuracy_score(targets, predicted)[0]
                precision = precision_score(targets, predicted, average='macro')[0]
                recall = recall_score(targets, predicted, average='macro')[0]
                f1 = f1_score(targets, predicted, average='macro')[0]
            except (IndexError, TypeError):
                accuracy = accuracy_score(targets, predicted)
                precision = precision_score(targets, predicted, average='macro')
                recall = recall_score(targets, predicted, average='macro')
                f1 = f1_score(targets, predicted, average='macro')
        loss = self.loss_val.item()
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc if self.classes == 2 else None,
            "fpr": fpr if self.classes == 2 else None,
            "loss": loss
        }

    def forward_pass(self, data):
        utils.reset(self.network)
        spk_list = []
        mem_list = []
        spk_out = None
        mem_out = None
        for _ in range(self.num_steps):
            if spk_out is not None and self.neuron in ['rsynaptic', 'lapicque']:
                utils.reset(self.network)
            if self.neuron_name in ['alpha']:
                spk_out, _, _, mem_out = self.network(data)
            elif self.neuron_name in ['rsynaptic', 'synaptic']:
                spk_out, _, mem_out = self.network(data)
            elif self.neuron_name in ['leaky', 'rleaky', 'lapicque']:
                spk_out, mem_out = self.network(data)
            else:
                raise NotImplementedError(f"Neuron {self.neuron_name} not implemented")
            spk_list.append(spk_out)
            mem_list.append(mem_out)
        spk_rec = torch.stack(spk_list)
        mem_rec = torch.stack(mem_list)
                
        return spk_rec, mem_rec
    
    def _get_device(self, gpu_number: int):
        """Get the device to use for the model.

        Args:
            gpu_number (int): GPU number to use.

        Returns:
            torch.device: Device to use.
        """
        if torch.cuda.is_available() and gpu_number != "cpu":
            try:
                print(f"Selected GPU {gpu_number}.")
                return torch.device(f"cuda:{gpu_number}")
            except Exception:
                print("Using default GPU.")
                return torch.device("cuda")
        elif torch.backends.mps.is_available() and gpu_number != "cpu":
            print("Using MPS.")
            return torch.device("mps")
        else:
            print("No GPU available, using CPU.")
            return torch.device("cpu")

    def _get_spiking_neuron(self, neuron: str, output=False, learn=True, **kwargs):
        self.neuron_name = neuron
        if neuron == 'lapicque':
            self.neuron_alpha = kwargs.get('neuron_alpha', None)
            self.neuron_beta = torch.tensor(kwargs.get('neuron_beta', 0.25), dtype=torch.float64, device=self.device)
            self.neuron_threshold = torch.tensor(kwargs.get('neuron_threshold', 1.0), dtype=torch.float64, device=self.device)
            self.neuron_recurrent = kwargs.get('neuron_V', None)
            return Lapicque(
                beta=self.neuron_beta, 
                threshold=self.neuron_threshold, 
                spike_grad=fast_sigmoid(slope=25), 
                init_hidden=True, 
                output=output, 
                learn_beta=learn, 
                learn_threshold=learn
            ).to(self.device)
        if neuron == 'alpha':
            self.neuron_alpha = kwargs.get('neuron_alpha', 0.9)
            self.neuron_beta = kwargs.get('neuron_beta', 0.8)
            self.neuron_threshold = kwargs.get('neuron_threshold', 1.0)
            self.neuron_recurrent = kwargs.get('neuron_V', None)
            return Alpha(
                alpha=self.neuron_alpha, 
                beta=self.neuron_beta, 
                threshold=self.neuron_threshold, 
                spike_grad=fast_sigmoid(slope=25), 
                init_hidden=True, 
                output=output, 
                learn_alpha=learn, 
                learn_beta=learn, 
                learn_threshold=learn
            )
        if neuron == 'leaky':
            self.neuron_alpha = kwargs.get('neuron_alpha', None)
            self.neuron_beta = kwargs.get('neuron_beta', 0.95)
            self.neuron_threshold = kwargs.get('neuron_threshold', 1.0)
            self.neuron_recurrent = kwargs.get('neuron_V', None)
            return Leaky(
                beta=self.neuron_beta, 
                threshold=self.neuron_threshold, 
                spike_grad=fast_sigmoid(slope=25), 
                init_hidden=True, 
                output=output, 
                learn_beta=learn, 
                learn_threshold=learn
            )
        if neuron == 'rleaky':
            self.neuron_alpha = kwargs.get('neuron_alpha', None)
            self.neuron_beta = kwargs.get('neuron_beta', 0.95)
            self.neuron_threshold = kwargs.get('neuron_threshold', 1.0)
            self.neuron_recurrent = kwargs.get('neuron_V', 0.5)
            return RLeaky(
                beta=self.neuron_beta, 
                threshold=self.neuron_threshold, 
                V=self.neuron_recurrent, 
                spike_grad=fast_sigmoid(slope=25), 
                init_hidden=True, 
                output=output, 
                learn_beta=learn, 
                learn_threshold=learn, 
                learn_recurrent=learn,
                all_to_all=False
            )
        if neuron == 'synaptic':
            self.neuron_alpha = kwargs.get('neuron_alpha', 0.5)
            self.neuron_beta = kwargs.get('neuron_beta', 0.95)
            self.neuron_threshold = kwargs.get('neuron_threshold', 1.0)
            self.neuron_recurrent = kwargs.get('neuron_V', None)
            return Synaptic(
                alpha=self.neuron_alpha, 
                beta=self.neuron_beta, 
                threshold=self.neuron_threshold, 
                spike_grad=fast_sigmoid(slope=25), 
                init_hidden=True, 
                output=output, 
                learn_alpha=learn, 
                learn_beta=learn, 
                learn_threshold=learn
            )
        if neuron == 'rsynaptic':
            self.neuron_alpha = kwargs.get('neuron_alpha', 0.5)
            self.neuron_beta = kwargs.get('neuron_beta', 0.95)
            self.neuron_threshold = kwargs.get('neuron_threshold', 1.0)
            self.neuron_recurrent = kwargs.get('neuron_V', 1.0)
            return RSynaptic(
                alpha=self.neuron_alpha, 
                beta=self.neuron_beta, 
                threshold=self.neuron_threshold, 
                V=self.neuron_recurrent, 
                spike_grad=fast_sigmoid(slope=25), 
                init_hidden=True, 
                output=output, 
                learn_alpha=learn, 
                learn_beta=learn, 
                learn_threshold=learn, 
                learn_recurrent=learn,
                all_to_all=False         
            ).to(self.device)
        raise ValueError(f"Neuron {neuron} not found, please try ['alpha', 'lapicque', 'leaky', 'rleaky', 'synaptic', 'rsynaptic']")
    
    def _get_network(self, neuron_name, classes, **kwargs):
        if self.dataset not in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'baf', 'unsw_nb15']:
            raise ValueError("Dataset not recognized") 
        if self.dataset in ['baf', 'unsw_nb15']:
            last_layer = 448 if self.dataset == 'baf' else 576
            return Sequential(
                Conv1d(1, 16, 2),
                MaxPool1d(2),
                self._get_spiking_neuron(neuron_name, output=False, **kwargs),
                Conv1d(16, 64, 2),
                MaxPool1d(2),
                self._get_spiking_neuron(neuron_name, output=False, **kwargs),
                Flatten(),
                Linear(last_layer, classes),
                self._get_spiking_neuron(neuron_name, output=True, **kwargs)
            )
        else:
            kernel_size = 5
            channels = 1 if 'cifar' not in self.dataset else 3
            output_layer = 64*4*4 if 'cifar' not in self.dataset else 64*5*5
            return Sequential(
                Conv2d(channels, 16, kernel_size),
                MaxPool2d(2),
                self._get_spiking_neuron(neuron_name, output=False, **kwargs),
                Conv2d(16, 64, 5),
                MaxPool2d(2),
                self._get_spiking_neuron(neuron_name, output=False, **kwargs),
                Flatten(),
                Linear(output_layer, classes),
                self._get_spiking_neuron(neuron_name, output=True, **kwargs)
            )