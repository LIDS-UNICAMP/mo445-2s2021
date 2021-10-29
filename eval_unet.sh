if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <number of the dataset>" >&2
  exit 1
fi

python src/eval_unet.py -a arch-unet.json -id datasets/dataset$1/imgs -gd datasets/dataset$1/gts -o output/
