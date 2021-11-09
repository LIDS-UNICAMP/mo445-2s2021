if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <image number>" >&2
  exit 1
fi

python src/extract_features.py -a arch-unet.json -i imgs_and_markers/$1.png -o features/ -m encoder.pt

libmo445/bin/dynamictrees imgs_and_markers/$1.png features/block2.mimg imgs_and_markers/$1.txt labels

