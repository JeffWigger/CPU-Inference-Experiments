# Infrastructure Deployment

Create a file called `basic.priv.tfvars` and add the following lines:
```
ssh_key_location      = "~/.ssh/your_key"
vm_admin_username     = "you_name"
subscription_id       = "your_subscription_id"
source_address_prefix = "your_current_ip"
```

Run the following command to provision the Azure VM (Standard_F8s_v2) on which the experiments were run:
(Note: the VM will cost you around 360$ per month)
```
az login
terraform init
terraform fmt
terraform plan -var-file=basic.priv.tfvars
terraform apply -var-file=basic.priv.tfvars
```

One the Linux machine run the following commands to set-up Poetry.

```
sudo apt-get update
sudo apt install build-essential
sudo apt-get install python-is-python
curl -sSL https://install.python-poetry.org | python3 -
```
