from pathlib import Path
import flammkuchen as fl


master = Path(r"J:\_Shared\experiments\E0022_multistim\v01_ephys\ECs")
data_list = []
for fish in master.glob("[0-1]*"):
    print(fish)
    data = dict()
    for session in ["exp022", "blanks", "lag"]:
        f = fish / session / "processed_spikes.h5"
        data[session] = None
        try:
            spikes = fl.load(str(f), group="/data_dict/spikes")
            twitches = fl.load(str(f), group="/data_dict/twitches")
            df = fl.load(str(f), group="/data_dict/traces")
            metadata = fl.load(str(f), group="/metadata")
            fn = metadata["fn"]
            data[session] = dict(
                spk_idxs=spikes,
                twc_idxs=twitches,
                spk_t=df.loc[spikes, "time"].values,
                spk_sweep=df.loc[spikes, "sweep"].values,
                twc_t=df.loc[twitches, "time"].values,
                twc_sweep=df.loc[twitches, "sweep"].values,
                fn=fn,
                name=f.parent.parent.name,
            )
        except OSError:
            print("No {} for this cell".format(session))
    data_list.append(data)

fl.save(master / "all_pooled_data.hdf5", data_list)
