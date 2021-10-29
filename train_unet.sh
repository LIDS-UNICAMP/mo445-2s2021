if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <num. of epochs>" >&2
  exit 1
fi

python src/train_flim_unet.py -a arch-unet.json -id imgs_and_markers/ -gd labels/ -ne $1
