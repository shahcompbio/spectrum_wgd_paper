docker run --platform linux/x86_64 --rm -p 8788:8787 -e DISABLE_AUTH=true \
    -v `pwd`:`pwd` -v `pwd`/packages:/user-library -v `pwd`:/home/rstudio \
    -v $SPECTRUM_PROJECT_DIR:$SPECTRUM_PROJECT_DIR -e SPECTRUM_PROJECT_DIR=$SPECTRUM_PROJECT_DIR \
    -w `pwd` amcpherson/gbm_longitudinal_cultures:latest
