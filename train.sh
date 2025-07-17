#!/user/bin/bash

for TDSBN in True False; do
    for TDSLN in True False; do
        for RNNBN in True False; do
            for RNNLN in True False; do
                python3 -m emg2qwerty.train \
                user="single_user" trainer.accelerator=gpu \
                trainer.devices=1 \
                module.batch_norm=$TDSBN module.layer_norm=$TDSLN \
                module.rnn_config.batch_norm=$RNNBN \
                module.rnn_config.layer_norm=$RNNLN;
            done;
        done;
    done;
done