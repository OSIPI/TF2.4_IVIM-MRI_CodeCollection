# Kaapana Installation Guide

## Overview

The Kaapana installation process consists of three main steps:

1. **Building** all the container images required for Kaapana and pushing them to a remote registry, local registry, or local tarball
2. **Installation** of Kaapana after building the images
3. **Deployment** of Kaapana after installation

For my setup, I used the remote registry approach, which is also the recommended format.

## Machine Requirements

These are the machine requirements which I used:

### System Specifications
- **Operating System**: Ubuntu 24.04
- **Platform**: Google Cloud Platform (GCP)

### VM Configuration
- **vCPU**: 8
- **RAM**: 64 GB
- **Disk**: 250 GB balanced persistent disk

## Step 1: Building Container Images

### Registry Setup (GitLab)

For the build process, I used GitLab registry. You'll need to configure three key parameters in your build-configuration file:

#### 1. `default_registry`
- Create a repository on GitLab
- Navigate to **Deploy** → **Container Registry**
- Copy the registry link (format: `registry.gitlab.com/your-username/repository-name`)
- Example: `registry.gitlab.com/unique-usman/kaapana-tutorial`

#### 2. `registry_password`
- Go to your repository **Settings** → **Access Tokens**
- Create a new token with read and write registry permissions
- *Note: I granted all permissions for simplicity*

#### 3. `registry_username`
- Use the GitLab username under which you created the repository
- Since I created my repository under my personal GitLab account, I used my GitLab username

### Build Process

Follow the comprehensive build guide available at:
https://kaapana.readthedocs.io/en/stable/installation_guide/build.html

## Step 2: Server Installation

### Installation Process

Refer to the server installation documentation:
https://kaapana.readthedocs.io/en/stable/installation_guide/server_installation.html

Since I didn't use any proxy configuration, I simply executed the server installation script as provided in the documentation.

## Step 3: Platform Deployment

### Deployment Process

Follow the deployment guide:
https://kaapana.readthedocs.io/en/stable/installation_guide/deployment.html

I ran the deployment script directly, and it worked perfectly without any additional configuration.

## Post-Deployment Configuration

### Accessing Your Kaapana Instance

After successful deployment, you'll receive a web link to access your Kaapana instance. 

### DNS Configuration

To simplify access, I mapped the deployment link to the server's IP address in `/etc/hosts`:

```bash
# Example entry in /etc/hosts
your-server-ip    your-kaapana-domain
```

## Architecture Notes

- **Single Machine Setup**: I used only one GCP machine for the entire Kaapana deployment
- **All-in-One Configuration**: The build, installation, and deployment processes were all executed on the same machine

## Key Resources

- [Build Guide](https://kaapana.readthedocs.io/en/stable/installation_guide/build.html)
- [Server Installation](https://kaapana.readthedocs.io/en/stable/installation_guide/server_installation.html)
- [Deployment Guide](https://kaapana.readthedocs.io/en/stable/installation_guide/deployment.html)

---

*This guide reflects a successful single-machine Kaapana deployment using GCP and GitLab registry.*
