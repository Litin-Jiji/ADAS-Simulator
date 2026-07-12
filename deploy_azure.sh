#!/bin/bash
# deploy_azure.sh
# One-time Azure setup script.
# Run this once to create all Azure resources.
# After this, GitHub Actions handles all future deploys.
#
# Prerequisites:
#   az login
#   az extension add --name containerapp

set -e

# ── Config — change these ──────────────────────────────────────────────────
RESOURCE_GROUP="adas-rg"
LOCATION="eastus"
ACR_NAME="adasregistry$(date +%s)"   # must be globally unique
ENVIRONMENT="adas-env"
# ──────────────────────────────────────────────────────────────────────────

echo "==> Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "==> Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

echo "==> Getting ACR credentials..."
ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USER=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

echo "==> Building and pushing images to ACR..."
az acr login --name $ACR_NAME

docker build -f Dockerfile.backend -t $ACR_SERVER/adas-backend:latest .
docker push $ACR_SERVER/adas-backend:latest

docker build -f Dockerfile.frontend -t $ACR_SERVER/adas-frontend:latest .
docker push $ACR_SERVER/adas-frontend:latest

echo "==> Creating Container Apps environment..."
az containerapp env create \
  --name $ENVIRONMENT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

echo "==> Deploying backend Container App..."
az containerapp create \
  --name adas-backend \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_SERVER/adas-backend:latest \
  --registry-server $ACR_SERVER \
  --registry-username $ACR_USER \
  --registry-password $ACR_PASS \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --env-vars DEVICE=cpu

BACKEND_URL=$(az containerapp show \
  --name adas-backend \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "==> Backend deployed at: https://$BACKEND_URL"

echo "==> Deploying frontend Container App..."
az containerapp create \
  --name adas-frontend \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_SERVER/adas-frontend:latest \
  --registry-server $ACR_SERVER \
  --registry-username $ACR_USER \
  --registry-password $ACR_PASS \
  --target-port 80 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 2

FRONTEND_URL=$(az containerapp show \
  --name adas-frontend \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "============================================="
echo "  ADAS Simulator deployed successfully!"
echo "============================================="
echo "  Frontend : https://$FRONTEND_URL"
echo "  Backend  : https://$BACKEND_URL"
echo "  ACR      : $ACR_SERVER"
echo "============================================="
echo ""
echo "Add these as GitHub Secrets for CI/CD:"
echo "  ACR_LOGIN_SERVER = $ACR_SERVER"
echo "  ACR_USERNAME     = $ACR_USER"
echo "  ACR_PASSWORD     = $ACR_PASS"
echo "  ACR_NAME         = $ACR_NAME"
