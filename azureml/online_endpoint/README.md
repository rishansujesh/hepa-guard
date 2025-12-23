# Azure ML Online Endpoint (HepaGuard)

## Deploy

```bash
az ml model create --name hepaguard-artifacts --path models
az ml online-endpoint create -f azureml/online_endpoint/endpoint.yml
az ml online-deployment create -f azureml/online_endpoint/deployment.yml
az ml online-endpoint update --name <endpoint> --traffic "<deployment>=100"
```

## Get keys

```bash
az ml online-endpoint get-credentials --name <endpoint>
```

## Invoke

```bash
az ml online-endpoint invoke --name <endpoint> --deployment <deployment> --request-file samples/core_high.json
```

```bash
curl -X POST "https://<endpoint>.inference.ml.azure.com/score" \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  --data @samples/core_high.json
```
