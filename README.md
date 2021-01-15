Update AWS cli to V2
Login to ECR for deep learning docker image
Build new image
Download TFRecord data from S3
Download resnet weights
Train model
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
alias aws2="/usr/local/aws-cli/v2/current/bin/aws"
aws2 ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
git clone -b awsdet_wip https://github.com/johnbensnyder/DeepLearningExamples/
mkdir -p ~/data/coco
pushd ~/data/coco
aws s3 cp --recursive s3://jbsnyder-sagemaker/data/coco/nv_coco/ .
popd
pushd DeepLearningExamples/scripts
./download_and_process_pretrained_weights.sh
./train_single_node.sh
```

