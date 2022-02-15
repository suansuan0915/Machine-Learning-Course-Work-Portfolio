# Pretrained Model Fine-Tuning

## Command to run my code

I run my code in **Google Colab**.

Colab *Command* running the .py file is like:

```
from google.colab import drive
drive.mount('/content/drive')
# !ls "/content/drive/My Drive/"

!python3 "/content/drive/My Drive/cifar_finetune.py"

```

## Implementation

For details about steps, please refer to report.pdf file.

- For L2 regularization

I add "weight_decay" hyperparameter in optimizer. 

- For validation

I save trained and get a model on trainset (5000 images), then load that model to do validation on testset (10000 images) every epoch.
Then specify a path and save the best model (the one with lowest validation loss) to that path by command:
```
PATH = '/content/drive/My Drive/train_model'
torch.save(model.state_dict(), PATH)

```

- Result

After tuning different hyperparameters, I found that when **"weight_decay"** in optimizer equals to 1e-3 can make the model reach a relatively lowest validation error of 0.971. Therefore, this is our best model (relatively).