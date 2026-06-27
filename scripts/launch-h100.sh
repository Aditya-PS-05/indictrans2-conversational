#!/usr/bin/env bash
# launch-h100.sh — find capacity for a single-H100 p5.4xlarge and launch it.
# Auto-discovers the Deep Learning OSS AMI + a SSH security group, then sweeps AZs.
# Usage: ./launch-h100.sh            # on-demand
#        SPOT=1 ./launch-h100.sh     # spot (~half price, interruptible)
set -uo pipefail

REGION="us-east-1"
ITYPE="p5.4xlarge"
KEY="gpu-key"
DISK_GB=300
SG_NAME="h100-train-sg"

echo "==> Discovering Deep Learning OSS AMI (supports P5)..."
# find the latest OSS PyTorch Ubuntu DLAMI parameter, then resolve to an AMI id
PARAM=$(aws ssm get-parameters-by-path --region "$REGION" \
  --path "/aws/service/deeplearning/ami/x86_64/" --recursive \
  --query "Parameters[?contains(Name,'oss-nvidia-driver-gpu-pytorch') && contains(Name,'ubuntu')].Name" \
  --output text | tr '\t' '\n' | sort | tail -n1)
if [ -z "${PARAM:-}" ]; then echo "!! Could not find an OSS DLAMI SSM param"; exit 1; fi
AMI=$(aws ssm get-parameter --region "$REGION" --name "$PARAM" --query Parameter.Value --output text)
echo "    AMI = $AMI  (from $PARAM)"

echo "==> Finding or creating a security group that allows SSH from your IP..."
MYIP=$(curl -s https://checkip.amazonaws.com)
SG=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)
if [ "$SG" = "None" ] || [ -z "$SG" ]; then
  VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text)
  SG=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$SG_NAME" --description "H100 training SSH" --vpc-id "$VPC" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG" --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null
  echo "    created SG $SG (SSH from ${MYIP}/32)"
else
  echo "    using existing SG $SG"
fi

# spot option
MARKET=""
if [ "${SPOT:-0}" = "1" ]; then
  MARKET='--instance-market-options {"MarketType":"spot"}'
  echo "==> SPOT mode enabled"
fi

SWEEP() {
echo "==> Sweeping AZs for $ITYPE capacity..."
for AZ in us-east-1a us-east-1b us-east-1c us-east-1d us-east-1f; do
  echo "--- trying $AZ"
  OUT=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$ITYPE" \
    --key-name "$KEY" --security-group-ids "$SG" \
    --placement "AvailabilityZone=$AZ" --count 1 \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=h100-train}]' \
    $MARKET 2>&1)
  if echo "$OUT" | grep -q '"InstanceId"'; then
    ID=$(echo "$OUT" | grep -o 'i-[0-9a-f]*' | head -n1)
    echo "==> SUCCESS: launched $ID in $AZ"
    echo "    Get IP: aws ec2 describe-instances --region $REGION --instance-ids $ID \\"
    echo "            --query 'Reservations[].Instances[].PublicIpAddress' --output text"
    echo "    SSH:    ssh -i ${KEY}.pem ubuntu@<ip>"
    exit 0
  elif echo "$OUT" | grep -qi 'InsufficientInstanceCapacity'; then
    echo "    no capacity in $AZ"
  else
    echo "    error in $AZ:"; echo "$OUT" | tail -n3
  fi
done
return 1   # no AZ had capacity this pass
}

# --- driver: single pass, or retry loop when LOOP=1 (interval via EVERY=secs) ---
if [ "${LOOP:-0}" = "1" ]; then
  EVERY="${EVERY:-45}"
  echo "==> LOOP mode: retrying every ${EVERY}s until capacity appears (Ctrl-C to stop)"
  TRY=0
  while true; do
    TRY=$((TRY+1)); echo "===== attempt $TRY ====="
    SWEEP && exit 0
    echo "    no capacity anywhere; sleeping ${EVERY}s..."
    sleep "$EVERY"
  done
else
  SWEEP && exit 0
  echo
  echo "==> All AZs are out of $ITYPE capacity (on-demand${SPOT:+/spot})."
  echo "    H100 is frequently dry. Options:"
  echo "      1. LOOP=1 ./launch-h100.sh        # auto-retry until a slot opens"
  echo "      2. SPOT=1 LOOP=1 ./launch-h100.sh # spot + auto-retry (~half price)"
  echo "      3. EC2 Capacity Blocks for ML     # reserve guaranteed time (most reliable)"
  exit 1
fi
