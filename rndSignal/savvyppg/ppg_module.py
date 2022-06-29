import numpy as np
import pandas as pd
import os
import pickle
from itertools import groupby, repeat
from multiprocessing import Pool, cpu_count
from rndSignal.savvyppg import ppg_utils as pu

# TODO: other fixed values should also be declared

# load the pickle file containing the good
# and bad beat classifier if there is any
base_path = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base_path, "models/randomforest_pca.pkl")
if os.path.isfile(filepath):
    dict_res = pd.read_pickle(filepath)
    classifier = dict_res["classifier"]
    comp = dict_res["comp"]
    pca = dict_res["pca"]
else:
    classifier, comp, pca = None, None, None

def wrapper_multiprocess(iterables, args):
    #http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma?rq=1
    #first element of args is the function to parallelize
    #the rest of the arguments are the other arguments aside from the iterable
    return args[0](iterables, args[1:])

cpu_cnt = cpu_count() #will determine the number of parallel processes
num_part = 10 #number of partitions

class ppgSignal():
    def __init__(self, data, fs, timestamp=pd.Timestamp("2000-1-1"),
                 hpass_cor_freq_hz=0.5, lpass_cor_freq_hz=3, cycle_method='1',
                 cycle_index=None, predict_beats=False, pca=pca, comp=comp,
                 classifier=classifier, filter_signal = True, cascade = False):
        self.data = np.array(data)
        self.fs = fs
        self._timestamp = timestamp
        self._hpass_cor_freq_hz = hpass_cor_freq_hz
        self._lpass_cor_freq_hz = lpass_cor_freq_hz

        #preprocess the data
        self.df = pu.preprocess(self.data, self.fs, self._hpass_cor_freq_hz,
                                self._lpass_cor_freq_hz, self._timestamp,
                                cycle_method=cycle_method, cycle_index=cycle_index, 
                                filter_signal = filter_signal,cascade = cascade)
        self.data_ppg = self.df["PPG"].values
        self.data_fdppg = self.df["FDPPG"].values
        self.data_sdppg = self.df["SDPPG"].values

        #option to predict good and bad beats
        if predict_beats:
            if classifier is not None:
                self.predict_good_bad_beats(pca,comp,classifier)

    def compute_features(self, pca=pca, comp=comp, classifier=classifier):
        #TODO: heart rate and heart rate variability must have signal
        #quality requirement
        #heart rate and heart rate variability
        try:
            self.beats_df["pred_label"]
        except (KeyError, AttributeError):
            self.predict_good_bad_beats(pca, comp, classifier)
        self.feat_systolic_index = pu.systolic_peak_ppg(self.data_ppg, self.fs)[0]
        #self.feat_systolic_bpm, self.feat_systolic_hrv, self.feat_systolic_rms, _ = \
        #    pu.systolic_to_systolic_bpm(self.feat_systolic_index, self.fs)
        self.feat_systolic_bpm, self.feat_systolic_hrv, self.feat_systolic_rms = \
            pu.heart_rate_good_beats(self.beats_df, self.fs)

        #respiration rate
        #run a signal quality measure before computing the respiration rate
        #make sure all the ppg beats are good beats, otherwise don't compute
        #for respiration rate
        if np.mean(self.beats_df["pred_label"])==1:
            self.feat_resp_ivsmooth,\
            self.feat_resp_fvsmooth,\
            self.feat_resp_avsmooth,\
            self.feat_breaths_per_min  = pu.respiration_rate_riv(self.data, self.fs)
        else:
            self.feat_resp_ivsmooth = np.nan
            self.feat_resp_fvsmooth = np.nan
            self.feat_resp_avsmooth = np.nan
            self.feat_breaths_per_min  = np.nan


        #oxygen saturation
        self.feat_trough_index = pu.trough_index_ppg(self.data_ppg,
                                                     self.feat_systolic_index)

        self.feat_systolic_envelope = pu.ppg_envelope(self.data_ppg,
                                                      self.feat_systolic_index)
        self.feat_trough_envelope = pu.ppg_envelope(self.data_ppg,
                                                    self.feat_trough_index)

        self.feat_hat = pu.ppg_envelope(self.data, self.feat_systolic_index)
        self.feat_shoe = pu.ppg_envelope(self.data, self.feat_trough_index)
        self.feat_dc_array = np.mean([self.feat_shoe, self.feat_hat], axis = 0)
        self.feat_ac_array = self.feat_systolic_envelope - self.feat_trough_envelope
        self.feat_ac_dc = np.mean(self.feat_ac_array)/np.mean(self.feat_dc_array)

        #blood pressure
        self.feat_bp_sysuptime = pu.systolic_upstroke_time(self.df, self.fs)
        self.feat_bp_diastime = pu.diastolic_time(self.df, self.fs)
        self.feat_bp_sysdiastime = pu.time_delay(self.df, self.fs)



        #other features

        self.feat_ppg_envelope_dc = np.mean([self.feat_systolic_envelope,
                                        self.feat_trough_envelope], axis=0)
        return self

    def compute_sdppgpeaksratios(self, method='1'):
        #TODO
        def remove_cycleid_info(df):
            if "cycle_id" in df.index.names:
                df = df.reset_index("cycle_id", drop = True)
            return df
        df = remove_cycleid_info(self.df)

        if method=='1':
            (peaks, ratios) = pu.peaksloc_ratios_sdppg1(df, self.fs)
        elif method=='2':
            (peaks, ratios) = pu.peaksloc_ratios_sdppg2(df, self.fs)
        else:
            raise ValueError("Invalid method")


        self.sdppg_idx_a, self.sdppg_idx_b, self.sdppg_idx_c,\
            self.sdppg_idx_d, self.sdppg_idx_e, self.sdppg_idx_f = peaks

        self.sdppg_ratio_ba, self.sdppg_ratio_ca, self.sdppg_ratio_da,\
            self.sdppg_ratio_ea, self.sdppg_ratio_fa = ratios

        return self

    def compute_ppgbeats(self, length=64, spline=3, level=False, scale="standardize",
                         time_idx = False):
        self.beats_df = pu.normalize_cycle_length(self.df, self.fs, len_new=length,
                        spline=spline, level=level, scale=scale, time_idx=time_idx)
        self.beats_num = len(self.beats_df)
        return self
    def compute_beat_level_features(self):
        #beat-level features: morphological features
        #these features will be used in predicting good or bad beats
        #and will be appended in the self.beats_df attribute

        col_names = ["morpfeat_diffheights_troughtopeak_ratio",
                     "morpfeat_amplitudes_ratio",
                     "morpfeat_duration_ratio",
                     "morpfeat_max_corr_coeff",
                     "morpfeat_corr_coeff_pearsonsr",
                     "morpfeat_variance_ratio",
                     "morpfeat_starttopeak_slope_ratio",
                     "morpfeat_peaktoend_slope_ratio",
                     "morpfeat_amplitude_duration_ratio",
                     "morpfeat_beat_area_ratio",
                     "morpfeat_energy_ratio"]
        if not hasattr(self, "beats_df"):
            #compute for the beats
            self.compute_ppgbeats()

        self.beats_df[col_names] = \
            pd.DataFrame([pu.diffheights_troughtopeak_ratio(self.df),
                          pu.amplitudes_ratio(self.df),
                          pu.duration_ratio(self.df),
                          pu.max_corr_coeff(self.df),
                          pu.corr_coeff_pearsonsr(self.df),
                          pu.variance_ratio(self.df),
                          pu.starttopeak_slope_ratio(self.df),
                          pu.peaktoend_slope_ratio(self.df),
                          pu.amplitude_duration_ratio(self.df),
                          pu.beat_area_ratio(self.df),
                          pu.energy_ratio(self.df)]).T
        return self
    def predict_good_bad_beats(self, pca=pca, comp=comp, classifier=classifier):
        #pca-pca model, comp-number of components, classifier-model
        #self.beats_df: cols and "PPG_beat_scale"
        #should run compute_ppgbeats, compute_beat_level_features
        cols = ["morpfeat_amplitude_duration_ratio",
                "morpfeat_amplitudes_ratio",
                "morpfeat_beat_area_ratio",
                "morpfeat_corr_coeff_pearsonsr",
                "morpfeat_diffheights_troughtopeak_ratio",
                "morpfeat_duration_ratio",
                "morpfeat_energy_ratio",
                "morpfeat_peaktoend_slope_ratio",
                "morpfeat_starttopeak_slope_ratio",
                "morpfeat_variance_ratio"]

        if not hasattr(self, "beats_df"):
            #computing for the beats
            self.compute_ppgbeats()

        try:
            self.beats_df[cols]
        except KeyError:
            #compute for the beat level features
            self.compute_beat_level_features()

        valid_inds = [x for x in self.beats_df.index                              \
                        if (~pd.isnull(self.beats_df.ix[x][cols]).any()&          \
                            ~pd.isnull(self.beats_df.ix[x]["PPG_beat_scale"]).any())]
        morpfeats = self.beats_df.ix[valid_inds][cols].values
        pca_feats = pca.transform(                                                \
                        np.vstack(self.beats_df.ix[valid_inds]["PPG_beat_scale"]) \
                                  )[:, :comp]
        preds = classifier.predict(np.concatenate((pca_feats, morpfeats), axis=1))
        self.beats_df["pred_label"] = np.array([np.nan]*len(self.beats_df))
        self.beats_df.loc[valid_inds, "pred_label"] = preds
        return self


class Plethysmogram():
    def __init__(self, ir_data, red_data, fs, timestamp=pd.Timestamp("2000-1-1"),
                 hpass_cor_freq_hz=0.5, lpass_cor_freq_hz=3, cycle_method='1',
                 predict_beats=False, pca=pca, comp=comp, classifier=classifier):
        self.fs = fs
        self._timestamp = timestamp
        self.ir = ppgSignal(ir_data, self.fs, timestamp=self._timestamp,
                    hpass_cor_freq_hz=hpass_cor_freq_hz,
                    lpass_cor_freq_hz=lpass_cor_freq_hz, cycle_method=cycle_method,
                    predict_beats=predict_beats, pca=pca, comp=comp,
                    classifier=classifier)
        cycle_index = self.ir.df.index.get_level_values(0)
        self.red = ppgSignal(red_data, self.fs, timestamp=self._timestamp,
                    hpass_cor_freq_hz=hpass_cor_freq_hz,
                    lpass_cor_freq_hz=lpass_cor_freq_hz, cycle_index=cycle_index,
                    predict_beats=predict_beats, pca=pca, comp=comp,
                    classifier=classifier)
    def compute_features(self, spo2_calibcurve=None, pca=pca,
                         comp=comp, classifier=classifier):

        #compute features for each signal
        self.ir.compute_features()
        self.red.compute_features()


        #compute spo2 in beat level(beat-based)
        #first get the intersection of good beats in the two channels
        #red and infrared signals, then compute for mean spo2 value

        #check if self.ir's and self.red's compute_ppgbeats(),
        #compute_beat_level_features() and predict_good_bad_beats()
        #were executed.
        for sig in [self.ir, self.red]:
            try:
                sig.beats_df["pred_label"]
            except (KeyError, AttributeError):
                sig.predict_good_bad_beats(pca, comp, classifier)

        #get the indices of the beats in beats_df
        indices = self.ir.beats_df["index"]
        #consider only captures with successive good beats
        num = 20 #requirement of good beats
        bools_ = ((self.ir.beats_df["pred_label"]==1) &
                    (self.red.beats_df["pred_label"]==1))
        diffs = np.diff(np.arange(len(bools_))[bools_.values])
        indicator = (np.array([len(list(val)) \
                               for (key,val) in groupby(diffs, lambda x: x==1) \
                               if key])>num).any()
        indices_ = indices.ix[bools_]
        if indicator:
            spo2 = []
            for ind in indices_:
                #spo2 version 1
                #R derivation based on the envelope of beat signal

                #ir_ac = self.ir.feat_ac_array[ind[0]: ind[1]]
                #ir_dc = self.ir.feat_dc_array[ind[0]: ind[1]]
                #acdc_ir = np.mean(ir_ac)/np.mean(ir_dc)

                #red_ac = self.red.feat_ac_array[ind[0]: ind[1]]
                #red_dc = self.red.feat_dc_array[ind[0]: ind[1]]
                #acdc_red = np.mean(red_ac)/np.mean(red_dc)
                #spo2.append(pu.compute_spo2_from_acdcratio(acdc_ir, acdc_red,
                #                            calib_curve=spo2_calibcurve)

                #spo2 version 2
                #R derivation based on the diff of the peak value and the ave.
                #of the 2 end troughs

                ir_ac = np.max(self.ir.data_ppg[ind[0]: ind[1]]) - \
                        np.mean((self.ir.data_ppg[ind[0]: ind[1]][0],
                                self.ir.data_ppg[ind[0]: ind[1]][-1]))
                ir_dc = np.mean(self.ir.data[ind[0]: ind[1]])
                acdc_ir = ir_ac/ir_dc


                red_ac = np.max(self.red.data_ppg[ind[0]: ind[1]]) - \
                        np.mean((self.red.data_ppg[ind[0]: ind[1]][0],
                                self.red.data_ppg[ind[0]: ind[1]][-1]))
                red_dc = np.mean(self.red.data[ind[0]: ind[1]])
                acdc_red = red_ac/red_dc
                spo2.append(pu.compute_spo2_from_acdcratio(acdc_ir, acdc_red,
                                                calib_curve=spo2_calibcurve))
            self.feat_spo2 = np.mean(spo2) #mean spo2
        else:
            #cases where no good beats detected
            self.feat_spo2 = np.nan

        #use the IR signal for the other physiological features
        self.feat_heart_rate = self.ir.feat_systolic_bpm
        self.feat_heart_rate_var = self.ir.feat_systolic_hrv
        self.feat_resp_rate = self.ir.feat_breaths_per_min
        return self

    def compute_sdppgpeaksratios(self, method='1'):
        #TODO:
        #computes the peaks and ratios for each
        #of the Signal instance in Plethysmogram instance
        #method is either '1' or '2'

        self.ir.compute_sdppgpeaksratios(method = method)
        self.red.compute_sdppgpeaksratios(method = method)

        return self

    def compute_ppgbeats(self, length=64, spline=3, level=False, scale="standardize",
                         time_idx = False):
        #computes the ppg beats for each of the
        #Signal instance in Plethysmogram instance
        #length is integer

        self.ir.compute_ppgbeats(length=length, spline=spline, level=level,
                                 scale=scale, time_idx=time_idx)
        self.red.compute_ppgbeats(length=length, spline=spline, level=level,
                                  scale=scale, time_idx=time_idx)

        return self

    def compute_beat_level_features(self):
        #compute beat-level features for ir signal only, these will be used
        #in predicting good and bad beats

        self.ir.compute_beat_level_features()
        self.red.compute_beat_level_features()
        return self
    def predict_good_bad_beats(self, pca=pca, comp=comp, classifier=classifier):
        self.ir.predict_good_bad_beats(pca, comp, classifier)
        self.red.predict_good_bad_beats(pca, comp, classifier)
        return self


class PlethysmogramDF():

    def _pletdf_transform(df, args):
        (ir_col, red_col, fs_col, timestamp_col, hpass_cor_freq_hz,lpass_cor_freq_hz,
        cycle_method, predict_beats, pca, comp, classifier) = args
        return df.apply(
            lambda x: Plethysmogram(x[ir_col], x[red_col],
                                    x[fs_col], x[timestamp_col],
                                    hpass_cor_freq_hz, lpass_cor_freq_hz,
                                    cycle_method, predict_beats=predict_beats,
                                    pca=pca, comp=comp, classifier=classifier),
                        axis=1)

    def __init__(self, df_data, ir_col, red_col, fs_col, timestamp_col = None,
                 hpass_cor_freq_hz=0.5, lpass_cor_freq_hz=3, cycle_method='1',
                 predict_beats=False, pca=pca, comp=comp, classifier=classifier):
        self.df_data = df_data.copy()
        self._ir_col = ir_col
        self._red_col = red_col
        self._fs_col = fs_col
        if timestamp_col:
            self._timestamp_col = timestamp_col
        else:
            self.df_data["timestamp"] = pd.Timestamp("2000-1-1")
            self._timestamp_col = "timestamp"
        self.num_samples = len(df_data)

        work_processes = cpu_cnt
        #base the number of work processes to machine's cores
        partitions = self.df_data.groupby(self.df_data.index//num_part)
        args = (PlethysmogramDF._pletdf_transform,self._ir_col, self._red_col,
                self._fs_col, self._timestamp_col, hpass_cor_freq_hz,
                lpass_cor_freq_hz, cycle_method, predict_beats, pca, comp,
                classifier)

        with Pool(work_processes) as pool:
            result = pool.starmap(wrapper_multiprocess,
                              zip([df[1] for df in partitions], repeat(args)))
        self.samples = pd.concat(list(result))
        #initialize self.partitions from hereon
        self.partitions = self.samples.groupby(self.samples.index//num_part)

        col_names = ["data", "fs", "df", "ppg", "fdppg", "sdppg"]
        self.df_ir_res = self.samples.apply(lambda x: x.ir).apply(
            lambda x: pd.Series([x.data, x.fs, x.df, x.data_ppg,
                                 x.data_fdppg, x.data_sdppg],
                                index = [i+"_ir" for i in col_names]))

        self.df_red_res = self.samples.apply(lambda x: x.red).apply(
            lambda x: pd.Series([x.data, x.fs, x.df, x.data_ppg,
                                 x.data_fdppg, x.data_sdppg],
                                index = [i+"_red" for i in col_names]))
        if predict_beats:
            col_names = ["beats_df", "beats_num"]

            self.df_ir_res[[i+"_ir" for i in col_names]] = \
                self.samples.apply(lambda x: x.ir)         \
                            .apply(lambda x: pd.Series([x.beats_df, x.beats_num]))

            self.df_red_res[[i+"_red" for i in col_names]] = \
                self.samples.apply(lambda x: x.red)          \
                            .apply(lambda x: pd.Series([x.beats_df, x.beats_num]))


    def _dist_computefeatures(samples, args):
        (spo2_calibcurve, pca, comp, classifier) = args
        return samples.apply(lambda x: x.compute_features(spo2_calibcurve=spo2_calibcurve,
                                    pca=pca, comp=comp, classifier=classifier))

    def compute_features(self, spo2_calibcurve=None, pca=pca, comp=comp,
                         classifier=classifier):
        args = (PlethysmogramDF._dist_computefeatures, spo2_calibcurve, pca, comp,
                classifier)

        with Pool(cpu_cnt) as pool:
            self.samples = pd.concat(list(
                pool.starmap(wrapper_multiprocess,
                             zip([d[1] for d in self.partitions], repeat(args)))))

        col_names = ["systolic_index", "systolic_bpm", "systolic_hrv",
                     "systolic_rms", "trough_index", "hat", "shoe", "dc_array",
                     "ac_array", "ac/dc", "sysuptime", "diastime", "sysdiastime",
                     "systolic_envelope", "trough_envelope", "ppg_envelope_dc",
                     "resp_ivsmooth", "resp_fvsmooth", "resp_avsmooth",
                     "breaths_per_min", "beats_df", "beats_num"]
        self.df_ir_res[[i+"_ir" for i in col_names]] = \
            self.samples.apply(lambda x: x.ir).apply(lambda x:  \
                        pd.Series([x.feat_systolic_index, x.feat_systolic_bpm,
                        x.feat_systolic_hrv, x.feat_systolic_rms,
                        x.feat_trough_index,x.feat_hat, x.feat_shoe, x.feat_dc_array,
                        x.feat_ac_array, x.feat_ac_dc, x.feat_bp_sysuptime,
                        x.feat_bp_diastime, x.feat_bp_sysdiastime,
                        x.feat_systolic_envelope, x.feat_trough_envelope,
                        x.feat_ppg_envelope_dc, x.feat_resp_ivsmooth,
                        x.feat_resp_fvsmooth, x.feat_resp_avsmooth,
                        x.feat_breaths_per_min, x.beats_df, x.beats_num]))

        self.df_red_res[[i+"_red" for i in col_names]] = \
            self.samples.apply(lambda x: x.red).apply(lambda x:  \
                        pd.Series([x.feat_systolic_index, x.feat_systolic_bpm,
                        x.feat_systolic_hrv, x.feat_systolic_rms,
                        x.feat_trough_index,x.feat_hat, x.feat_shoe, x.feat_dc_array,
                        x.feat_ac_array, x.feat_ac_dc, x.feat_bp_sysuptime,
                        x.feat_bp_diastime, x.feat_bp_sysdiastime,
                        x.feat_systolic_envelope, x.feat_trough_envelope,
                        x.feat_ppg_envelope_dc, x.feat_resp_ivsmooth,
                        x.feat_resp_fvsmooth, x.feat_resp_avsmooth,
                        x.feat_breaths_per_min, x.beats_df, x.beats_num]))

        spo2 = self.samples.apply(lambda x: x.feat_spo2)
        #append it in self.df_ir_res and self.df_red_res
        for df in [self.df_ir_res, self.df_red_res]:
            df["spo2"] = spo2

        ind_names = ["heart_rate", "heart_rate_var", "resp_rate", "spo2"]
        self.physio_features = self.samples.apply(\
                                    lambda x: pd.Series([x.feat_heart_rate,
                                                         x.feat_heart_rate_var,
                                                         x.feat_resp_rate,
                                                         x.feat_spo2],
                                                        index = ind_names))
        return self


    def _dist_computesdppgpeaksratios(samples, args):
        method = args[0]
        return samples.apply(lambda x: x.compute_sdppgpeaksratios(method=method))

    def compute_sdppgpeaksratios(self, method='1'):
        args = (PlethysmogramDF._dist_computesdppgpeaksratios, method)
        with Pool(cpu_cnt) as pool:
            self.samples=pd.concat(list(
                            pool.starmap(wrapper_multiprocess,
                                         zip([d[1] for d in self.partitions],
                                             repeat(args)))))

        col_names = ["sdppg_idx_a", "sdppg_idx_b", "sdppg_idx_c", "sdppg_idx_d",
                     "sdppg_idx_e", "sdppg_idx_f", "sdppg_ratio_ba",
                     "sdppg_ratio_ca", "sdppg_ratio_da", "sdppg_ratio_ea",
                     "sdppg_ratio_fa"]
        self.df_ir_res[[i+"_ir" for i in col_names]] = \
            self.samples.apply(lambda x: x.ir).apply(lambda x: \
                    pd.Series([x.sdppg_idx_a, x.sdppg_idx_b, x.sdppg_idx_c,
                        x.sdppg_idx_d, x.sdppg_idx_e, x.sdppg_idx_f,
                        x.sdppg_ratio_ba, x.sdppg_ratio_ca, x.sdppg_ratio_da,
                        x.sdppg_ratio_ea, x.sdppg_ratio_fa]))

        self.df_red_res[[i+"_red" for i in col_names]] = \
            self.samples.apply(lambda x: x.red).apply(lambda x: \
                    pd.Series([x.sdppg_idx_a, x.sdppg_idx_b, x.sdppg_idx_c,
                        x.sdppg_idx_d, x.sdppg_idx_e, x.sdppg_idx_f,
                        x.sdppg_ratio_ba, x.sdppg_ratio_ca, x.sdppg_ratio_da,
                        x.sdppg_ratio_ea, x.sdppg_ratio_fa]))

        return self


    def _dist_computeppgbeats(samples, args):
        (length, spline, level, scale, time_idx) = args
        return samples.apply(lambda x: x.compute_ppgbeats(length=length, spline=spline,
                                    level=level, scale=scale, time_idx=time_idx))

    def compute_ppgbeats(self, length=64, spline=3, level=False, scale="standardize",
                         time_idx = False):

        args = (PlethysmogramDF._dist_computeppgbeats, length, spline, level,
                scale, time_idx)
        with Pool(cpu_cnt) as pool:
            self.samples=pd.concat(list(
                            pool.starmap(wrapper_multiprocess,
                                         zip([d[1] for d in self.partitions],
                                             repeat(args)))))

        col_names = ["beats_df", "beats_num"]

        self.df_ir_res[[i+"_ir" for i in col_names]] = \
            self.samples.apply(lambda x: x.ir).apply(lambda x: \
                    pd.Series([x.beats_df, x.beats_num]))

        self.df_red_res[[i+"_red" for i in col_names]] = \
            self.samples.apply(lambda x: x.red).apply(lambda x: \
                    pd.Series([x.beats_df, x.beats_num]))

        return self


    def _dist_computebeatlevelfeatures(samples, args):
        return samples.apply(lambda x: x.compute_beat_level_features())

    def compute_beat_level_features(self):

        args = (PlethysmogramDF._dist_computebeatlevelfeatures, None)
        with Pool(cpu_cnt) as pool:
            self.samples = pd.concat(list(
                            pool.starmap(wrapper_multiprocess,
                                         zip([d[1] for d in self.partitions],
                                              repeat(args)))))


        col_names = ["beats_df", "beats_num"]

        self.df_ir_res[[i+"_ir" for i in col_names]] = \
            self.samples.apply(lambda x: x.ir).apply(lambda x: \
                    pd.Series([x.beats_df, x.beats_num]))

        self.df_red_res[[i+"_red" for i in col_names]] = \
            self.samples.apply(lambda x: x.red).apply(lambda x: \
                    pd.Series([x.beats_df, x.beats_num]))

        return self


    def _dist_predictgoodbadbeats(samples, args):
        (pca, comp, classifier) = args
        return samples.apply(lambda x: x.predict_good_bad_beats(pca, comp,
                                                                classifier))
    def predict_good_bad_beats(self, pca=pca, comp=comp, classifier=classifier):

        args = (PlethysmogramDF._dist_predictgoodbadbeats, pca, comp, classifier)
        with Pool(cpu_cnt) as pool:
            self.samples = pd.concat(list(
                                pool.starmap(wrapper_multiprocess,
                                             zip([d[1] for d in self.partitions],
                                                  repeat(args)))))

        col_names = ["beats_df", "beats_num"]

        self.df_ir_res[[i+"_ir" for i in col_names]] = \
            self.samples.apply(lambda x: x.ir).apply(lambda x: \
                    pd.Series([x.beats_df, x.beats_num]))

        self.df_red_res[[i+"_red" for i in col_names]] = \
            self.samples.apply(lambda x: x.red).apply(lambda x: \
                    pd.Series([x.beats_df, x.beats_num]))

        return self


