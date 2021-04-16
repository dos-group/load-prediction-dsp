# Steps

Install environment:

    conda env create -f environment.yaml
  
Switch to environment:

    source activate ELPTforDSP
  
Start script, e.g.:

    python slim_run.py -drd ../data -dn alibaba avazu google horton IoT retailrocket taxi wiki_de wiki_en -dsr 5min 15min 1h -m GRU -d cuda:0 -cr 1 -gr 0.125 -mmn
