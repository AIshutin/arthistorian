# API server

Calls other services if necessary.


```shell
rm datamodel_server.zip
zip datamodel_server * && mv datamodel_server.zip ../datamodel_server.zip


PROJECT_ID=arthistorian
IMAGE=datepredictor_server

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE
```
