import pyxdf,re,pickle,mne,os,warnings,csv
import numpy as np
from dataclasses import dataclass
from scipy import signal
from modules.modwt import modwt, imodwt
from scipy.signal import find_peaks
from typing import Optional, List, Tuple, Dict

def get_montage(montage):
    if os.path.isfile(montage):
        montage_obj = mne.channels.read_custom_montage(montage)
    else:
        montage_obj = mne.channels.make_standard_montage(montage)
    return montage_obj

def bandpass_fir(x, fs, lo=None, hi=None, numtaps=None):
    if lo is None and hi is None:
        return x.copy()
    nyq = fs / 2.0
    if numtaps is None:
        numtaps = int(1.5 * fs)
        if numtaps % 2 == 0:
            numtaps += 1
    if lo is not None and hi is not None:
        b = signal.firwin(numtaps, [lo / nyq, hi / nyq], pass_zero=False)
    elif lo is not None:
        b = signal.firwin(numtaps, lo / nyq, pass_zero=False)
    else:
        b = signal.firwin(numtaps, hi / nyq)
    return signal.filtfilt(b, [1.0], x, axis=-1)
# Load XDF file and convert to RawData
def load_xdf_to_raw(xdf_path: str,
    rename_dict: Optional[Dict[str, str]] = None,
    ref_names: Tuple[str, ...] = ("M1", "M2")) -> "RawData":
    # 1. Load XDF file
    streams, header = pyxdf.load_xdf(xdf_path)
    # 2. Extract stream
    sfreqs = 512  # or: float(streams[0]['info']['effective_srate'])
    data = streams[0]["time_series"].T
    times = streams[0]["time_stamps"]
    ch_info = streams[0]["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = np.array([c["label"][0] for c in ch_info])
    # 3. Separate channels
    eog_data = data[np.isin(ch_names, ["EOG"])][0]
    ref_data = data[np.isin(ch_names, list(ref_names))]
    ecg_data = data[np.isin(ch_names, ["AUX7"])][0]
    resp_data = data[np.isin(ch_names, ["AUX8"])][0]
    trigger = data[np.isin(ch_names, ["TRIGGER"])][0]
    eeg_data = data[~np.isin(ch_names, ['EOG', 'M1', 'M2', 'AUX7', 'AUX8', 'TRIGGER'])]
    ch_names = ch_names[~np.isin(ch_names, ['EOG', 'M1', 'M2', 'AUX7', 'AUX8', 'TRIGGER'])].tolist()
    # 4. Rename channels (optional)
    if rename_dict is not None:
        ch_names = [rename_dict.get(ch, ch) for ch in ch_names]
    # 5. Extract events from trigger
    idx = np.nonzero(trigger)[0]
    vals = trigger[idx]
    events = np.vstack((idx, vals)).T.astype(int)

    # 6. Create RawData instance
    return RawData(eeg=eeg_data,eog=eog_data,ecg=ecg_data,
        resp=resp_data,ref=ref_data,trigger=trigger,times=times,
        events=events,ch_names=ch_names,sfreq=sfreqs,)
# Load File
def load(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj
# Load Triggers from CSV
def get_triggers_id(csv_path: str) -> List[List]:
    triggers_id = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)  
        for row in reader:
            triggers_id.append([row["label"],
                int(row["start"]),int(row["end"]),])
    return triggers_id
"""
def bandpass(x, fs, lo=None, hi=None, order=4, ftype='butter'):
    if lo is not None and hi is not None:
        wn = [2 * lo / fs, 2 * hi / fs]
        btype = 'band'
    elif lo is not None:
        wn = 2 * lo / fs
        btype = 'high'
    elif hi is not None:
        wn = 2 * hi / fs
        btype = 'low'
    else:
        return x.copy()
    b, a = signal.iirfilter(order, wn, btype=btype, ftype=ftype)
    return signal.filtfilt(b, a, x, axis=-1)
"""

@dataclass
class RawData:
    eeg: Optional[np.ndarray] = None       # (n_eeg_ch, n_times)
    eog: Optional[np.ndarray] = None       # (n_times)
    ecg: Optional[np.ndarray] = None       # (n_times,)
    resp: Optional[np.ndarray] = None      # (n_times,)
    ref: Optional[np.ndarray] = None       # (n_ch, n_times)
    trigger: Optional[np.ndarray] = None   # (n_times,)
    times: Optional[np.ndarray] = None     # (n_times,)

    events: Optional[np.ndarray] = None    # (n_events, 2) -> sample_idx, value
    ch_names: Optional[List[str]] = None   # EEG channel names
    sfreq: Optional[float] = None          # sampling rate Hz

    def copy(self) -> "RawData":
        return RawData(
            eeg=self.eeg.copy(),
            eog=None if self.eog is None else self.eog.copy(),
            ecg=None if self.ecg is None else self.ecg.copy(),
            resp=None if self.resp is None else self.resp.copy(),
            ref=None if self.ref is None else self.ref.copy(),
            trigger=None if self.trigger is None else self.trigger.copy(),
            times=None if self.times is None else self.times.copy(),
            events=None if self.events is None else self.events.copy(),
            sfreq=self.sfreq,ch_names=list(self.ch_names),
        )
    # Remove DC offset
    def remove_ecg_dc(self) -> "RawData":
        out = self.copy()
        ecg_mean = np.mean(out.ecg)
        out.ecg = out.ecg - ecg_mean
        return out
    
    def remove_resp_dc(self) -> "RawData":
        out = self.copy()
        resp_mean = np.mean(out.resp)
        out.resp = out.resp - resp_mean
        return out
    # Apply Rerefence
    def apply_reref(self, drop_ref: bool = True) -> "RawData":
        out = self.copy()
        if out.ref is None:
            warnings.warn(
            "apply_reref() was called but no reference channel (`ref`) is present. "
            "No re-referencing was applied. Please set `ref` before calling this method.",
            RuntimeWarning)
            return out
        ref_mean = out.ref.mean(axis=1, keepdims=True)
        out.eeg = out.eeg - ref_mean
        if drop_ref:
            out.ref = None
        return out
    # Interpolate Bad channels
    def interpolate_bads(self, bads=None, montage: str = "standard_1020") -> "RawData":
        out = self.copy()
        if bads is None:
            return out
        montage_obj = get_montage(montage)
        info = mne.create_info(ch_names=out.ch_names, sfreq=out.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(out.eeg, info)
        raw.set_montage(montage_obj)
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)
        out.eeg = raw.get_data()
        return out
    # Bandpass
    def filter(self, low=None, high=None, ch_type=None, numtaps=None) -> "RawData":
        out = self.copy()
        def _bp(arr): 
            return None if arr is None else bandpass_fir(arr, out.sfreq, lo=low, hi=high, numtaps=numtaps)
        #out.eeg, out.eog, out.ecg, out.ref, out.resp = map(_bp, (out.eeg, out.eog, out.ecg, out.ref, out.resp))
        if ch_type == None:
            ch_type = ["eeg", "eog", "ecg", "ref", "resp"]
        for key in ch_type:
            setattr(out, key, _bp(getattr(out, key)))
        return out
    # Segment by Event
    def segment_by_event(self, eventid: Tuple[int,int]) -> List["RawData"]:
        # Find event onsets and offsets
        events = self.events
        sta = events[events[:, 1] == eventid[0]]
        end = events[events[:, 1] == eventid[1]]
        # Segment for each type
        def _slice(arr: Optional[np.ndarray],
                 sta_idx: int,end_idx: Optional[int]):
            if arr is None:
                return None
            return arr[..., sta_idx:end_idx]
        # Extract All Segment
        segments: List[RawData] = []
        for ii in range(len(sta)):
            sta_idx = sta[ii][0]
            end_idx = end[ii][0]+1 if ii < len(end) else self.trigger.shape[-1]
            seg = self.copy()
            # slice all channels
            for key in ["eeg", "eog", "ref", "ecg", "resp","trigger","times"]:
                setattr(seg, key, _slice(getattr(self, key), sta_idx, end_idx))
            # slice events
            mask = (events[:, 0] >= sta_idx) & (events[:, 0] < end_idx)
            seg_events = events[mask].copy()
            # let segment from 0
            seg_events[:, 0] -= sta_idx
            seg.events = seg_events
            segments.append(seg)
        return segments
    # Detect Peaks and Transform to Events
    def detect_peaks_to_events(self,
        wavelet: str = "sym4", level: int = 5, keep_levels: tuple = (4, 5), 
        min_peak_distance_ms: int = 300, min_peak_height: Optional[float] = 300,
        ecg_bandpass: tuple = (0.1, 40), event_id: int = 9) -> np.ndarray:
        # 1. MODWT decomposition
        ecg = bandpass_fir(self.ecg, self.sfreq, lo=ecg_bandpass[0], hi=ecg_bandpass[1])
        wt = modwt(ecg, wavelet, level)
        # 2. Keep only specified levels
        wtrec = np.zeros_like(wt)
        for k in keep_levels:
            wtrec[k - 1, :] = wt[k - 1, :] 
        # 3. Inverse MODWT
        y = imodwt(wtrec, wavelet)
        # 4. Compute squared magnitude
        # y2 = np.abs(y) ** 2
        # 5. Detect peaks
        min_distance = int(self.sfreq * (min_peak_distance_ms / 1000.0))
        peaks, _ = find_peaks(y, distance=min_distance, height=min_peak_height)
        # 6. Adjust peak positions to original ECG signal
        peaks_adj = []
        for p in peaks:
            if p + 1 < len(ecg) and ecg[p + 1] > ecg[p]:
                peaks_adj.append(p + 1)
            else:
                peaks_adj.append(p)
        peaks_adj = np.array(peaks_adj, dtype=int)
        return np.column_stack((peaks_adj, np.full_like(peaks, event_id))).astype(int)
    # Add ecg peaks events
    def add_events(self,events:np.ndarray):
        out = self.copy()
        merged = np.vstack([out.events, events])
        out.events = merged[np.argsort(merged[:, 0])]
        return out
    # Detect ECG Artifacts
    def detect_ecg_artifacts(self, amp_thresh: float=10000.0,
                    max_gap_sec: float=800.0) -> "RawData":
        out = self.copy()
        if out.ecg is None or out.sfreq is None:
            return out
        # Identify bad segments where ECG amplitude exceeds threshold
        samples_per_gap = int(max_gap_sec * out.sfreq)
        idx_bad = np.where(np.abs(out.ecg) > amp_thresh)[0]
        # Split into continuous segments
        bad_segid = np.where(np.diff(idx_bad) > 1)[0]+1
        bad_seg = np.split(idx_bad, bad_segid)
        # Filter out short segments
        cleaned_seg = [seg for seg in bad_seg if len(seg) > samples_per_gap]
        bad_idx = np.concatenate(cleaned_seg) if len(cleaned_seg) > 1 else np.array([], dtype=int)
        # Set Bad time points to NaN
        if bad_idx.size:
            out.ecg[..., bad_idx] = np.nan
        return out
    # Extract Epochs
    def extract_epochs(self, time_window: Tuple[float, float],
                       event_id: int,label_list=None) -> "EpochsData":
        # Segment for each type
        def _slice(arr: Optional[np.ndarray],
                 sta_idx: int,end_idx: Optional[int]):
            return arr[..., sta_idx:end_idx]
        # Find events
        evts = np.asarray(self.events)
        sel = evts[:, 1] == event_id
        evtids = evts[sel][:,0]
        # Function of Combine Labels
        def _combine(labels,ss):
            ll_arr = np.array(labels, dtype=object)
            merged = np.array([ss.join([x for x in col if  x not in (None, "")]) 
                                for col in ll_arr.T])
            return merged
        # Set labels
        if label_list is None:
            labels = None
        else:
            llsall = []
            for ll_list in label_list:
                llsub_rows = []
                for ll, sta_id, end_id in ll_list:
                    sta = np.where(evts[:, 1] == sta_id)[0]
                    end = np.where(evts[:, 1] == end_id)[0]
                    lls = np.full(len(evts), None, dtype=object)
                    for s, e in zip(sta, np.append(end, None)):
                        lls[s + 1 : e] = ll
                    llsub_rows.append(lls)
                llsall.append( _combine(llsub_rows, '_'))
            llsall = _combine(llsall, '/')
            labels = llsall[sel]
        # Get Epochs
        sig_keys = ["eeg", "eog", "ecg", "ref", "resp"]
        sig_lists = {k: [] for k in sig_keys}
        for event in evtids:
            sta_time = event + int(time_window[0] * self.sfreq)
            end_time = event + int(time_window[1] * self.sfreq)
            for key in sig_keys:
                arr = getattr(self, key)
                if arr is not None:
                    sig_lists[key].append(_slice(arr, sta_time, end_time))
        # Reject Epochs without enough time points and with NaN values
        sig_lists["labels"] = labels
        sig_lists["events"] = self.events[sel] 
        shapes = [dd.shape for dd in sig_lists["eeg"]]
        target_shape = max(set(shapes), key=shapes.count)
        keep_idx = [ i for i in range(len(shapes))
            if shapes[i] == target_shape and not np.isnan(sig_lists["ecg"][i]).any()]
        stacked = {k: np.array([v[i] for i in keep_idx]) if len(v) > 0 else None
                   for k, v in sig_lists.items()}

        return EpochsData(eeg=stacked["eeg"], eog=stacked["eog"], ecg=stacked["ecg"],
                        resp=stacked["resp"], ref=stacked["ref"], labels=stacked["labels"],
                        events=stacked["events"],sfreq=self.sfreq,ch_names=self.ch_names,
                        tmin=time_window[0], tmax=time_window[1])
    # Save File
    def save(self, filepath: str):
        filepath = str(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

@dataclass
class EpochsData:
    eeg: Optional[np.ndarray] = None
    eog: Optional[np.ndarray] = None
    ecg: Optional[np.ndarray] = None
    resp: Optional[np.ndarray] = None
    ref: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None      # label
    events: Optional[np.ndarray] = None
    sfreq: Optional[float] = None
    ch_names: Optional[List[str]] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None

    @property
    def times(self) -> np.ndarray:
        return np.linspace(self.tmin, self.tmax, self.eeg.shape[-1], endpoint=False)

    def copy(self) -> "EpochsData":
        return EpochsData(
            eeg=self.eeg.copy(),
            eog=None if self.eog is None else self.eog.copy(),
            ecg=None if self.ecg is None else self.ecg.copy(),
            resp=None if self.resp is None else self.resp.copy(),
            ref=None if self.ref is None else self.ref.copy(),
            labels=None if self.labels is None else self.labels.copy(),
            events=None if self.events is None else self.events.copy(),
            sfreq=self.sfreq,
            ch_names=list(self.ch_names),
            tmin=self.tmin, tmax=self.tmax,
        )
    # Reject Bad Epochs
    def reject_bad_epochs(self, threshold: float = 100.0) -> "EpochsData":
        out = self.copy()
        abs_max = np.max(np.abs(out.eeg), axis=(1,2))
        bad_idx = np.where(abs_max > threshold)[0]
        if bad_idx.size:
            keep = np.setdiff1d(np.arange(out.eeg.shape[0]), bad_idx)
            out.eeg = out.eeg[keep]
            if out.eog is not None: out.eog = out.eog[keep]
            if out.ecg is not None: out.ecg = out.ecg[keep]
            if out.resp is not None: out.resp = out.resp[keep]
            if out.ref  is not None: out.ref  = out.ref[keep]
            if out.events  is not None: out.events  = out.events[keep]
            if out.labels  is not None: out.labels  = out.labels[keep]
        return out
    # Apply Rerefence
    def apply_reref(self, drop_ref: bool = True) -> "EpochsData":
        out = self.copy()
        if out.ref is None:
            return out
        ref_mean = out.ref.mean(axis=1, keepdims=True)
        out.eeg = out.eeg - ref_mean
        if drop_ref:
            out.ref = None
        return out
    # Apply Baseline
    def apply_baseline(self, bl_windows: Tuple[float,float] = None, mode: str = "mean") -> "EpochsData":
        out = self.copy()
        if bl_windows is None:
            if out.tmin >= 0:
                return out
            bl_windows = [out.tmin,0]
        bl_id = ((np.array(bl_windows) - out.tmin) * out.sfreq).astype(int)
        base_eeg = out.eeg[:,:,bl_id[0]:bl_id[1]].mean(axis=-1, keepdims=True)
        base_ref = out.ref[:,:,bl_id[0]:bl_id[1]].mean(axis=-1, keepdims=True)
        if mode == "mean":
            out.eeg = out.eeg - base_eeg
            out.ref = out.ref - base_ref
        elif mode == "zscore":
            std_eeg = out.eeg[:,:,bl_id[0]:bl_id[1]].std(axis=-1, keepdims=True) + 1e-12
            out.eeg = (out.eeg - base_eeg) / std_eeg
            std_ref = out.ref[:,:,bl_id[0]:bl_id[1]].std(axis=-1, keepdims=True) + 1e-12
            out.ref = (out.ref - base_ref) / std_ref
        return out
    # Select Epochs as Lables
    def select_epochs(self, target = None)-> "EpochsData":
        out = self.copy()
        sig_keys = ["eeg", "eog", "ecg", "ref", "resp","labels","events"]
        if target is None:
            return out
        split_labels = [re.split(r"[/_]", lab) if isinstance(lab, str) else []
                            for lab in self.labels]
        target_set = set(target) if isinstance(target, (list, tuple, set)) else {str(target)}
        mask = np.array([any(t in labs for t in target_set)
                            for labs in split_labels])
        for key in sig_keys:
            arr = getattr(out, key)
            arr = arr[mask] if arr is not None else None
            setattr(out, key, arr)
        return out
    #Average Epochs
    def average(self) -> "EvokedData":
        return EvokedData(
            eeg=None if self.eeg is None else self.eeg.mean(axis=0),
            eog=None if self.eog is None else self.eog.mean(axis=0),
            ecg=None if self.ecg is None else self.ecg.mean(axis=0),
            resp=None if self.resp is None else self.resp.mean(axis=0),
            ref=None if self.ref is None else self.ref.mean(axis=0),
            labels=self.labels,
            events=self.labels,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            tmin=self.tmin,
            tmax=self.tmax,
            times=self.times,
        )
    # Save File
    def save(self, filepath: str):
        filepath = str(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Transform To MNE Epochs
    def trans_to_mne_epochs(self, montage = None) -> mne.Epochs:
        info = mne.create_info(ch_names=self.ch_names,sfreq=self.sfreq,ch_types="eeg")
        mne_epochs = mne.EpochsArray(self.eeg, info, tmin=self.tmin)
        if montage is not None:
            montage_obj = get_montage(montage)
            mne_epochs.set_montage(montage_obj)
        if self.ecg is not None:
            info_ecg = mne.create_info(ch_names=["ECG"], sfreq=self.sfreq, ch_types=["ecg"])
            epochs_ecg = mne.EpochsArray(self.ecg[:, np.newaxis, :], info_ecg, tmin=self.tmin)
            mne_epochs = mne_epochs.add_channels([epochs_ecg])
        return mne_epochs
    # Run ICA
    def run_ica(self, n_components: int = 20, method: str = "fastica",
                 random_state: int = 97, plot: bool = True):
        mne_epochs = self.trans_to_mne_epochs()
        ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state)
        ica.fit(mne_epochs, picks="eeg")
        if plot:
            ica.plot_components()
            ica.plot_sources(mne_epochs)
        return ica
    # Auto detect ICA components to exclude (ECG + muscle)
    def auto_find_bad_ica(self,ica) -> list[int]:
        mne_epochs = self.trans_to_mne_epochs()
        # Detect ECG
        ecg_inds, _ = ica.find_bads_ecg(mne_epochs)
        # Detect EMG (muscle)
        muscle_inds, _ = ica.find_bads_muscle(mne_epochs)
        # Apply ICA cleaning
        return ecg_inds + muscle_inds
    # Exclude ICA components
    def exclude_ica(self,ica,exclude_list) -> "EpochsData":
        out = self.copy()
        mne_epochs = self.trans_to_mne_epochs()
        ica.exclude = exclude_list
        mne_epochs_clean = ica.apply(mne_epochs.copy())
        out.eeg = mne_epochs_clean.get_data(picks="eeg")
        #out.ica = ica
        return out

@dataclass
class EvokedData:
    eeg: Optional[np.ndarray] = None
    eog: Optional[np.ndarray] = None
    ecg: Optional[np.ndarray] = None
    resp: Optional[np.ndarray] = None
    ref: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None      # label
    events: Optional[np.ndarray] = None
    sfreq: Optional[float] = None
    ch_names: Optional[List[str]] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    times: Optional[np.ndarray] = None

    def trans_to_mne_evoked(self, montage=None, comment="") -> mne.Evoked:
        info = mne.create_info(ch_names=self.ch_names,sfreq=self.sfreq,ch_types="eeg")
        evoked = mne.EvokedArray(self.eeg, info, tmin=self.tmin, comment=comment)
        if montage is not None:
            montage_obj = get_montage(montage)
            evoked.set_montage(montage_obj)
        return evoked
    # Save File
    def save(self, filepath: str):
        filepath = str(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Load File
    def load(filepath: str) -> "EvokedData":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj
# Plot Epochs
def plot_chs(ax,epochs_list,ch_list=None,class_list=[None],stats=None):
    results = []
    for cc in class_list:
        class_sub = []
        for epochs in epochs_list:
            ep = epochs.select_epochs(cc)
            epdd = ep.eeg.mean(0)
            class_sub.append(epdd)
        results.append(class_sub)
    results = np.array(results)
    times = ep.times
    if ch_list is not None:
        ch_idx = [ep.ch_names.index(c) for c in ch_list]
        results = results[:,:,ch_idx].mean(2)
    for ii in range(len(class_list)):
        ax.plot(times,results[ii].mean(0),label=f"{class_list[ii]}")
    return ax