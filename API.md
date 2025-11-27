# HEP-EEG Toolbox – API Documentation

This document describes the public API of `core.py`, including utility functions and the core data classes:
`RawData`, `EpochsData`, and `EvokedData`.

---

## 1. Utility Functions

### `load_xdf_to_raw(xdf_path, rename_dict=None, ref_names=("M1", "M2"))`

Load an XDF file and convert it into a `RawData` object.

This function assumes a specific channel naming scheme where EEG, EOG, ECG, respiration, trigger, and reference channels are stored in the same stream and are separated based on their labels.

**Parameters**

- `xdf_path` (`str`): Path to the `.xdf` file.

- `rename_dict` (`dict[str, str]` or `None`):  
  Optional mapping from original EEG channel names to new names.  
  If `None`, original names are kept.

- `ref_names` (`tuple[str, ...]`):  
  Names of the reference channels (Defaults to `("M1", "M2")`.).

**Returns**

- `RawData`:  
  A `RawData` instance containing:
  - `eeg`: EEG channels (after excluding EOG, reference, AUX7, AUX8, TRIGGER)  
  - `eog`: EOG channel  
  - `ecg`: ECG channel (`"AUX7"`)  
  - `resp`: Respiration channel (`"AUX8"`)  
  - `ref`: Reference channels specified by `ref_names`  
  - `trigger`: Trigger channel (`"TRIGGER"`)  
  - `times`: Time stamps  
  - `events`: Event matrix derived from non-zero trigger samples  
  - `ch_names`: EEG channel names  
  - `sfreq`: Sampling frequency

---

### `get_montage(montage)`

Load a montage either from a file path or from a standard MNE montage name.

**Parameters**

- `montage` (`str`):  
  - If it is a valid file path, load with `mne.channels.read_custom_montage`.
  - Otherwise, treat as a standard montage name in MNE and load with `mne.channels.make_standard_montage`.

**Returns**

- `montage_obj`: MNE montage object.

---

### `bandpass_fir(x, fs, lo=None, hi=None, numtaps=None)`

FIR band-pass / high-pass / low-pass filtering using `scipy.signal.firwin` and `signal.filtfilt`.

**Parameters**

- `x` (`np.ndarray`): Input signal. The last dimension is assumed to be time, e.g.
  `(n_times,)`, `(n_ch, n_times)`, or `(n_epochs, n_ch, n_times)`.
- `fs` (`float`): Sampling frequency (Hz).
- `lo` (`float or None`): Low cutoff frequency (Hz). If `None` and `hi` is not `None`, performs low-pass.
- `hi` (`float or None`): High cutoff frequency (Hz). If `None` and `lo` is not `None`, performs high-pass.
- `numtaps` (`int or None`): Length of the FIR filter.  
  If `None`, defaults to `int(1.5 * fs)` and is forced to be odd.

**Returns**

- `np.ndarray`: Filtered signal with the same shape as `x`.

---

### `load(filepath)`

Load any HEP-EEG object from a pickle file.

**Parameters**

- `filepath` (`str`): Path to the file .

**Returns**

- `object`: The deserialized object, typically one of:
  - `RawData`
  - `EpochsData`
  - `EvokedData`
---

### `get_triggers_id(csv_path)`

Load trigger label definitions from a CSV file.  
Each row must contain the fields `"label"`, `"start"`, and `"end"`.
**CSV format example**

```csv
label,start,end
MPS,13,16
MPS,17,26
Control,33,36
Control,37,46
```
**Parameters**

- `csv_path` (`str`): Path to the file .

**Returns**

- `list[list]`: A list of `[label, start_id, end_id]`

---

## 2. `RawData` Class

Continuous EEG data container with optional EOG, ECG, respiration, reference, trigger, and event information.

### Attributes

- `eeg` (`np.ndarray or None`): EEG data, shape `(n_eeg_ch, n_times)`.
- `eog` (`np.ndarray or None`): EOG data, typically shape `(n_times,)`.
- `ecg` (`np.ndarray or None`): ECG data, typically shape `(n_times,)`.
- `resp` (`np.ndarray or None`): Respiration signal, typically shape `(n_times,)`.
- `ref` (`np.ndarray or None`): Reference channels, shape `(n_ref_ch, n_times)`.
- `trigger` (`np.ndarray or None`): Trigger signal, shape `(n_times,)`.
- `times` (`np.ndarray or None`): Time stamps, shape `(n_times,)`.
- `events` (`np.ndarray or None`): Event matrix of shape `(n_events, 2)`, each row `[sample_idx, value]`.
- `ch_names` (`List[str] or None`): EEG channel names.
- `sfreq` (`float or None`): Sampling frequency in Hz.

---

### `copy()`

Deep copy of the `RawData` object.

**Returns**

- `RawData`: A new object with copied arrays.

---

### `remove_ecg_dc()`

Remove DC offset from the ECG channel (subtract mean).

**Returns**

- `RawData`: New object with ECG DC removed.

---

### `remove_resp_dc()`

Remove DC offset from the respiration channel.

**Returns**

- `RawData`: New object with respiration DC removed.

---

### `apply_reref(drop_ref: bool = True)`

Apply average re-reference using the reference channels.

If no reference channel is present (`ref is None`), the function returns the object unchanged and issues a `RuntimeWarning`.

**Parameters**

- `drop_ref` (`bool`):  
  If `True`, set `ref` to `None` after re-referencing.  
  If `False`, keep `ref`.

**Returns**

- `RawData`: New object with re-referenced EEG.

---

### `interpolate_bads(bads=None, montage: str = "standard_1020")`

Interpolate bad EEG channels using MNE’s `Raw.interpolate_bads`.

**Parameters**

- `bads` (`list or None`): List of bad channel names. If `None`, no interpolation is performed.
- `montage` (`str`): Montage specification used to define channel positions.
  Can be a standard MNE montage name (e.g., `"standard_1020"`) or a file path to a custom `.loc` montage.


**Returns**

- `RawData`: New object with interpolated EEG.

---

### `filter(low=None, high=None, ch_type=None, numtaps=None)`

Apply FIR filtering to selected signal types.  
This method is a general wrapper around `bandpass_fir()` and supports band-pass,
low-pass, and high-pass filtering depending on the provided cutoff parameters.

**Parameters**

- `low` (`float or None`): Low cutoff frequency (Hz).  
  If `None` and `high` is provided → low-pass filter.

- `high` (`float or None`): High cutoff frequency (Hz).  
  If `None` and `low` is provided → high-pass filter.

- `ch_type` (`list[str] or None`): Names of signal attributes to filter (e.g., `["eeg", "ecg"]`).  
  If `None`, all signal types `["eeg", "eog", "ecg", "ref", "resp"]` are filtered.

- `numtaps` (`int or None`): Length of the FIR filter.  
  If `None`, defaults to `int(1.5 * fs)` (adjusted to an odd number).

**Returns**

- `RawData`:  
  A new `RawData` object with the specified channels filtered.


---

### `segment_by_event(eventid: Tuple[int, int])`

Segment the continuous data into multiple `RawData` segments based on start and end event types.

**Parameters**

- `eventid` (`tuple(int, int)`): `(start_code, end_code)` specifying the event values used as segment boundaries.

**Returns**

- `List[RawData]`: List of segmented `RawData` objects.
  Each segment has sliced signals, `times`, and `events` (re-indexed to start at 0).

---

### `detect_peaks_to_events(wavelet="sym4", level=5, keep_levels=(4,5), min_peak_distance_ms=300, min_peak_height=300, ecg_bandpass=(0.1,40), event_id=9)`

Detect ECG R-peaks using MODWT and return them as an event matrix.

**Parameters**

- `wavelet` (`str`, default `"sym4"`): Wavelet type for MODWT.
- `level` (`int`, default `5`): Decomposition level.
- `keep_levels` (`tuple`, default `(4, 5)`): Detail levels to keep for reconstruction.
- `min_peak_distance_ms` (`int`, default `300`): Minimum peak distance in milliseconds.
- `min_peak_height` (`float or None`, default `300`): Minimum peak height.
- `ecg_bandpass` (`tuple`, default `(0.1, 40)`): Band-pass range for ECG before peak detection.
- `event_id` (`int`, default `9`): Event code assigned to detected peaks.

**Returns**

- `np.ndarray`: Array of shape `(n_peaks, 2)`, each row `[sample_idx, event_id]`.

---

### `add_events(events: np.ndarray)`

Merge externally provided events (e.g., ECG peaks) into the existing event matrix and sort by time.

**Parameters**

- `events` (`np.ndarray`): Event array of shape `(n_events, 2)`.

**Returns**

- `RawData`: New object with merged events.

---

### `detect_ecg_artifacts(amp_thresh: float = 10000.0, max_gap_sec: float = 800.0)`

Detect segments of ECG data with abnormally large amplitude and mark them as artifacts.

**Parameters**

- `amp_thresh` (`float`): Amplitude threshold; samples with `|ECG| > amp_thresh` are considered candidate artifacts.
- `max_gap_sec` (`float`): Minimum duration (in seconds) for a segment to be kept as artifact.

**Returns**

- `RawData`: New object where artifact samples in `ecg` are set to `NaN`.

---

### `extract_epochs(time_window: Tuple[float, float], event_id: int, label_list=None)`

Extract epochs locked to a given event type.

**Parameters**

- `time_window` (`tuple(float, float)`): Time window relative to the event, e.g. `(-0.2, 1.0)` in seconds.
- `event_id` (`int`): Event code to lock epochs to.
- `label_list` (`list or None`): Advanced labeling rules (list of `(label, start_id, end_id)` tuples) used to build epoch labels from the event stream.  
  If `None`, `labels` will be `None`.

**Returns**

- `EpochsData`: Epoch structure containing EEG/EOG/ECG/resp/ref, labels, events, and metadata.

---

### `save(filepath: str)`

Save the `RawData` object using `pickle`.

**Parameters**

- `filepath` (`str`): Output file path.

---

## 3. `EpochsData` Class

Epoch-based EEG data structure (trial × channel × time).

### Attributes

- `eeg` (`np.ndarray or None`): Shape `(n_epochs, n_ch, n_times)`.
- `eog` (`np.ndarray or None`): Shape `(n_epochs, n_times)`.
- `ecg` (`np.ndarray or None`): Shape `(n_epochs, n_times)`.
- `resp` (`np.ndarray or None`): Shape `(n_epochs, n_times)`.
- `ref` (`np.ndarray or None`): Shape `(n_epochs, n_ch, n_times)`.
- `labels` (`np.ndarray or None`): Labels per epoch.
- `events` (`np.ndarray or None`): Event info for each epoch.
- `sfreq` (`float or None`): Sampling frequency.
- `ch_names` (`List[str] or None`): Channel names.
- `tmin` (`float or None`): Epoch start time (seconds).
- `tmax` (`float or None`): Epoch end time (seconds).

---

### `times` (property) -> np.ndarray

Computed time axis for the epoch.

**Returns**

- `np.ndarray`: Time vector from `tmin` to `tmax` (exclusive), length equals `eeg.shape[-1]`.

---

### `copy()`

Deep copy of the `EpochsData` object.

---

### `reject_bad_epochs(threshold: float = 100.0)`

Reject epochs based on absolute EEG amplitude.

**Parameters**

- `threshold` (`float`): Epochs with `max(|eeg|)` greater than `threshold` are removed.

**Returns**

- `EpochsData`: New object with only "good" epochs.

---

### `apply_reref(drop_ref: bool = True)`

Apply average reference at the epoch level.

**Parameters**

- `drop_ref` (`bool`): If `True`, set `ref` to `None` after re-referencing.

**Returns**

- `EpochsData`

---

### `apply_baseline(bl_windows: Tuple[float, float] = None, mode: str = "mean")`

Baseline correction or baseline-based z-scoring.

**Parameters**

- `bl_windows` (`tuple(float, float) or None`):  
  Baseline window (in seconds) relative to epoch time.  
  If `None` and `tmin < 0`, defaults to `(tmin, 0)`.  
  If `tmin >= 0`, no correction is applied.
- `mode` (`str`):  
  - `"mean"`: Subtract the mean of the baseline interval.  
  - `"zscore"`: Subtract mean and divide by standard deviation of the baseline interval.

**Returns**

- `EpochsData`: Baseline-corrected (and optionally z-scored) epochs.

---

### `select_epochs(target=None)`

Select a subset of epochs based on `labels`.

The function splits each label by `/` and `_`, and checks if any target token is present.

**Parameters**

- `target` (`str or list or set or tuple or None`):  
  - If `None`, returns all epochs.  
  - If string or list/set/tuple of strings, only epochs whose label tokens contain any of the targets are kept.

**Returns**

- `EpochsData`: New object with selected epochs.

---

### `average()`

Compute the average across all epochs for each channel and signal type.

**Returns**

- `EvokedData`: Evoked response (mean over epochs).

---

### `save(filepath: str)`

Save the `EpochsData` object using `pickle`.

**Parameters**

- `filepath` (`str`): Output file path.

---

### `trans_to_mne_epochs(montage=None)`

Convert the current `EpochsData` object into an MNE `EpochsArray`.  
EEG channels are always included. If an ECG signal is available, it is
converted into a separate MNE channel and appended using `add_channels`.

**Parameters**

- `montage` (`str or None`):  
  Optional montage specification.
  If `None`, no montage is applied.

**Returns**

- `mne.Epochs`:  
  An MNE `EpochsArray` containing:  
  - EEG channels (with optional montage applied)  
  - An additional `"ECG"` channel if ECG data is present  

---

### `run_ica(n_components: int = 20, method: str = "fastica", plot: bool = True)`

Run ICA on the MNE epochs representation of the data.

**Parameters**

- `n_components` (`int`): Number of ICA components.
- `method` (`str`): ICA method (Defaults to `"fastica"`).
- `random_state` (`int`): Seed for reproducible ICA initialization. Defaults to `97`.
- `plot` (`bool`): If `True`, plot ICA components and sources.

**Returns**

- `mne.preprocessing.ICA`: Fitted ICA object.

---

### `auto_find_bad_ica(ica)`

Automatically detect ICA components associated with common physiological artifacts.

This method analyzes the current data using:
- `ica.find_bads_ecg()` for ECG-related components  
- `ica.find_bads_muscle()` for muscle (EMG) artifacts  

It returns the combined list of unique artifact components **without applying**
the ICA cleaning. Use `exclude_ica()` to actually remove them.

**Parameters**

- `ica` (`ICA`):  
  Fitted ICA object obtained from `run_ica()`.

**Returns**

- `list[int]`:  
  List of ICA component indices suggested for exclusion.

---

### `exclude_ica(ica, exclude_list)`

Apply the ICA solution to remove specified components and return cleaned epochs.

**Parameters**

- `ica` (`mne.preprocessing.ICA`): Fitted ICA object.
- `exclude_list` (`list[int]`): Indices of components to exclude.

**Returns**

- `EpochsData`: New object with cleaned EEG.

---

## 4. `EvokedData` Class

Average (evoked) data structure.

### Attributes

- `eeg` (`np.ndarray or None`): Evoked EEG, shape `(n_ch, n_times)`.
- `eog` (`np.ndarray or None`).
- `ecg` (`np.ndarray or None`).
- `resp` (`np.ndarray or None`).
- `ref` (`np.ndarray or None`).
- `labels` (`np.ndarray or None`).
- `events` (`np.ndarray or None`).
- `sfreq` (`float or None`).
- `ch_names` (`List[str] or None`).
- `tmin` (`float or None`).
- `tmax` (`float or None`).
- `times` (`np.ndarray or None`): Time vector.

---

### `trans_to_mne_evoked(montage=None, comment: str = "")`

Convert the evoked EEG data into an MNE `EvokedArray`.

**Parameters**

- `montage` (`str or None`):  
  Optional montage specification.
  If `None`, no montage is applied.
- `comment` (`str`): Comment string stored in the MNE object.

**Returns**

- `mne.Evoked`: MNE evoked object.

---

### `save(filepath: str)`

Save the `EvokedData` object using `pickle`.

**Parameters**

- `filepath` (`str`): Output file path.

---

## 5. Plotting
> **Note**  
> The plotting utilities will be expanded in future
---
