"""Explicit weighted ALP batches and globally normalized NumPy export.

Normalize the fragmentation group's raw weights once over ALL batches at
one mass and variation. Native/exclusive owners retain unit weight. Apply
the resulting hadronization weights in addition to physical decay weights.
Never normalize each chunk separately or treat the raw records as unweighted.
"""
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import argparse
import io
import json
import math
from pathlib import Path
import tempfile
import numpy as np
from runtime_generator import EventBatch, RuntimeEventGenerator, _legacy_random_seed


@dataclass(frozen=True)
class WeightedEventBatch(EventBatch):
    raw_hadronization_weights: np.ndarray
    hadronization_groups: np.ndarray
    normalization_key: tuple

    def event(self, position):
        raise TypeError('Use weighted batch records and explicit hadronization weights; '
                        'the unweighted EventRecord format cannot represent this sample')


class WeightedALPGenerator(RuntimeEventGenerator):
    """Opt-in ALP importance sampling; requires exHad rc8 or later."""

    def __init__(self, *args, weight_floor_fraction=0., **kwargs):
        super().__init__(*args, **kwargs)
        if (isinstance(weight_floor_fraction, bool) or not np.isfinite(weight_floor_fraction)
                or not 0. <= weight_floor_fraction <= 1.):
            raise ValueError('weight_floor_fraction must lie in [0, 1]')
        self.weight_floor_fraction = weight_floor_fraction

    def init(self):
        result = super().init()
        binding = getattr(self.llp, '_exhad_binding', None)
        if binding is None or binding['model'] != 'alp-fermion':
            raise ValueError('WeightedALPGenerator requires the ALP exHad model')
        from funcs.exhad_integration import _generator
        if not hasattr(_generator(binding),'generate_weighted'):
            raise ValueError('Weighted generation requires exHad rc8 or later')
        return result

    def _decay(self, mother, momentum):
        decay_seed = self._block_seeds(2, self._decay_block, count=1)[0]
        self._decay_block += 1
        output = nullcontext() if self.verbose else redirect_stdout(io.StringIO())
        with _legacy_random_seed(decay_seed, seed_numba=True), output:
            from funcs.exhad_integration import simulate_decays
            rest, sizes, metadata = simulate_decays(self.llp, self.runtime.decayProducts,
                self.llp.mass, self.llp.PDGs, self.llp.BrRatios_distr, len(mother),
                self.llp.Matrix_elements, list(self.selected_decay_indices),
                self.visible_branching_ratio, seed=decay_seed, weighted=True,
                weight_floor_fraction=self.weight_floor_fraction)
        boosted = self.runtime.boost.tab_boosted_decay_products(
            self.llp.mass, momentum, np.asarray(rest, dtype=float))
        from funcs.exhad_integration import matched_process_labels
        labels = matched_process_labels(self.llp.decayChannels,
                                        self.selected_decay_indices,
                                        self.llp._exhad_pooled_indices)
        names = np.asarray([(str(label) if label is not None
                             else str(self.llp.decayChannels[index]))
                            for index, size, label in zip(
                                self.selected_decay_indices, sizes, labels, strict=True)
                            for _ in range(int(size))], dtype=str)
        if len(names) != len(mother):
            raise RuntimeError('Weighted event/channel alignment failed')
        self._hadronization_metadata = metadata
        return boosted, names

    def _make_batch(self, count):
        base = super()._make_batch(count)
        return WeightedEventBatch(**base.__dict__,
            raw_hadronization_weights=self._hadronization_metadata['raw_weights'],
            hadronization_groups=self._hadronization_metadata['normalization_groups'],
            normalization_key=('alp-fermion',self.mass,'central',self.weight_floor_fraction,
                               self.llp._exhad_binding['version']))


class HadronizationNormalization:
    """Accumulate ALL batches at one mass/configuration before normalization."""

    def __init__(self):
        self.key=None
        self.events=0
        self.fragmentation_events=0
        self._sums=[]
        self._squares=[]

    def add(self,batch):
        if self.key is not None and batch.normalization_key != self.key:
            raise ValueError('Cannot mix masses, variations, floors or releases in one normalization')
        weights=np.asarray(batch.raw_hadronization_weights,dtype=float)
        groups=np.asarray(batch.hadronization_groups)
        if weights.shape != (len(batch),) or groups.shape != weights.shape:
            raise ValueError('Weight metadata does not align with the batch')
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('Importance weights must be finite and positive')
        if not set(groups).issubset({'external','fragmentation'}):
            raise ValueError('Unknown normalization group')
        active=groups=='fragmentation'
        if np.any(weights[~active] != 1.):
            raise ValueError('External-owner weights must be one')
        self.key=batch.normalization_key
        self.events+=len(batch)
        self.fragmentation_events+=int(active.sum())
        self._sums.append(float(weights[active].sum()))
        self._squares.append(float(np.square(weights[active]).sum()))

    def as_dict(self):
        if not self.events:
            raise ValueError('No events to normalize')
        total=math.fsum(self._sums)
        factor=self.fragmentation_events/total if self.fragmentation_events else 1.
        denominator=self.events-self.fragmentation_events+factor**2*math.fsum(self._squares)
        return dict(events=self.events,fragmentation_events=self.fragmentation_events,
            raw_fragmentation_sum=total,fragmentation_factor=factor,
            hadronization_effective_sample_size=self.events**2/denominator,
            normalization_key=list(self.key))


def export_weighted(generator,events,output_dir,*,batch_size=10000):
    """Write bounded batches, applying ONE full-sample normalization at the end.

    Final NPZs contain separate physical and hadronization weights, and their
    product as ``analysis_weights``. Temporary raw files are not final output.
    """
    if isinstance(events,bool) or not isinstance(events,int) or events<=0:
        raise ValueError('events must be a positive integer')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<=0:
        raise ValueError('batch_size must be a positive integer')
    generator.init()
    output_dir=Path(output_dir).resolve()
    output_dir.mkdir(parents=True,exist_ok=False)
    accumulator=HadronizationNormalization();files=[]
    with tempfile.TemporaryDirectory(prefix='.weighted-raw-',dir=output_dir) as temporary:
        raw_files=[]
        for batch in generator.generate(events,batch_size=batch_size):
            accumulator.add(batch)
            name=f'events_{batch.start:012d}_{batch.stop:012d}.npz'
            path=Path(temporary)/name
            np.savez_compressed(path,records=batch.records,channels=batch.channels,
                inside_volume=batch.inside_volume,decay_weights=batch.decay_weights,
                raw_hadronization_weights=batch.raw_hadronization_weights,
                hadronization_groups=batch.hadronization_groups,start=batch.start,stop=batch.stop)
            raw_files.append(path)
        normalization=accumulator.as_dict()
        for path in raw_files:
            with np.load(path,allow_pickle=False) as archive:
                values={name:archive[name] for name in archive.files}
            weights=values['raw_hadronization_weights'].copy()
            weights[values['hadronization_groups']=='fragmentation']*=normalization['fragmentation_factor']
            values['hadronization_weights']=weights
            values['analysis_weights']=weights*values['decay_weights']
            np.savez_compressed(output_dir/path.name,**values)
            files.append(path.name)
    from funcs.exhad_integration import metadata
    report=dict(format='EventCalc weighted runtime batches v1',model='ALP-fermion',
        mass_GeV=generator.mass,c_tau_m=generator.c_tau,events=events,batch_size=batch_size,
        mode=generator.mode,seed=generator.resolved_seed,
        hadronization=metadata(generator.llp),weight_floor_fraction=generator.weight_floor_fraction,
        normalization=normalization,experiment=generator.experiment.as_dict(),
        weights={'decay_weights':'Physical decay probabilities, unchanged',
            'raw_hadronization_weights':'Unnormalized importance weights; use normalized values below',
            'hadronization_weights':'Globally normalized across all batches in this output',
            'analysis_weights':'decay_weights multiplied by hadronization_weights'},
        normalization_scope='Full generated sample, before any daughter selection; one mass and configuration',
        yields=generator.yields().as_dict(),files=files)
    with (output_dir/'metadata.json').open('x') as stream:json.dump(report,stream,indent=2)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',type=Path,required=True)
    parser.add_argument('--mass',type=float,required=True)
    parser.add_argument('--ctau',type=float,required=True)
    parser.add_argument('--events',type=int,required=True)
    parser.add_argument('--batch-size',type=int,default=10000)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--weight-floor-fraction',type=float,default=.1)
    parser.add_argument('--mode',choices=['fiducial','attempted'],default='fiducial')
    parser.add_argument('--experiment-card',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    from funcs.simulation_config import PROJECT_ROOT,config_from_mapping,load_card
    from runtime_generator import ExperimentCard
    values=dict(load_card(args.card))
    values.update(masses=[args.mass],c_taus=[args.ctau],events=args.events,exhad_mode='on')
    if args.seed is not None:values['seed']=args.seed
    config=config_from_mapping(values,project_root=PROJECT_ROOT)
    experiment=ExperimentCard.load(args.experiment_card) if args.experiment_card else ExperimentCard.ship()
    generator=WeightedALPGenerator(config,mass=args.mass,c_tau=args.ctau,mode=args.mode,
        experiment=experiment,weight_floor_fraction=args.weight_floor_fraction)
    report=export_weighted(generator,args.events,args.output_dir,batch_size=args.batch_size)
    print(json.dumps(dict(events=report['events'],normalization=report['normalization'],
                         output=str(args.output_dir.resolve()))))


if __name__=='__main__':main()
