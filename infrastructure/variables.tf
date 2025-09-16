#    Copyright 2025 Jeffrey Wigger
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

variable "ssh_key_location" {
  default     = "~/.ssh/id_rsa.pub"
  description = "Path to the SSH key"
}

variable "rg_location" {
  default     = "switzerlandnorth"
  description = "Azure Resource Group location"
}

variable "vm_machine_size" {
  default     = "Standard_F8s_v2"
  description = "VM machine size"
}

variable "vm_admin_username" {
  default     = "azureuser"
  description = "Name of the root user"
}

variable "source_address_prefix" {
  default     = "*"
  description = "Allowed sources for SSH inbound connections"
}

variable "subscription_id" {
  description = "Azure subscription id"
}
