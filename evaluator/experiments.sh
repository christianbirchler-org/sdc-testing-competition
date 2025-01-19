#!/bin/bash

SENSODAT_COLLECTIONS=('campaign_10_ambiegen' 'campaign_11_ambiegen' 'campaign_11_frenetic' 'campaign_11_frenetic_v' 'campaign_12_frenetic' 'campaign_12_freneticV' 'campaign_13_ambiegen' 'campaign_13_frenetic' 'campaign_13_frenetic_v' 'campaign_14_ambiegen' 'campaign_14_frenetic' 'campaign_14_frenetic_v' 'campaign_15_ambiegen' 'campaign_15_frenetic' 'campaign_15_freneticV' 'campaign_2_ambiegen' 'campaign_2_frenetic' 'campaign_2_frenetic_v' 'campaign_3_ambiegen' 'campaign_3_frenetic' 'campaign_4_ambiegen' 'campaign_4_frenetic' 'campaign_4_frenetic_v' 'campaign_5_ambiegen' 'campaign_5_frenetic' 'campaign_5_frenetic_v' 'campaign_6_ambiegen' 'campaign_6_frenetic' 'campaign_6_frenetic_v' 'campaign_7_ambiegen' 'campaign_7_frenetic' 'campaign_7_frenetic_v' 'campaign_8_ambiegen' 'campaign_8_frenetic' 'campaign_9_ambiegen' 'campaign_9_frenetic')

for col in "${SENSODAT_COLLECTIONS[@]}"; do
    echo "$col"
    docker run -p 4545 --gpus all --network=host --rm "$1" -u localhost:4545 -c "$col"
done
