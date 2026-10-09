"""
Description
-----------
Tests of the NCMC protocol bookkeeping in the nonequilibrium samplers: custom lambda grids,
seeding, per-move records and per-step work traces. Uses a small water-free box of benzenes on the
Reference platform, so runs are exactly reproducible and random insertions cannot clash (the
Reference platform can crash, rather than raise, on blown-up coordinates).
"""

import logging
import os

import numpy as np
import pytest
from openmm import Platform, Vec3
from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile, Simulation
from openmm.unit import (
    Quantity,
    is_quantity,
    kelvin,
    kilocalories_per_mole,
    nanometers,
    picosecond,
    picoseconds,
)
from openmmtools.integrators import BAOABIntegrator

from grandlig import samplers, utils

BENZENE_PDB = utils.get_data_file(os.path.join("tests", "Benzene.pdb"))
BENZENE_XML = utils.get_data_file(os.path.join("tests", "Benzene.xml"))
TEMPERATURE = 298 * kelvin
DT = 0.002 * picoseconds
LAMBDAS = [0.0, 0.2, 0.5, 0.75, 0.9, 1.0]


@pytest.fixture(scope="module")
def benzene_box():
    """A 2.6 nm periodic box (no solvent) with one coupled benzene (L01) at its centre and two ghost copies."""
    pdb = PDBFile(BENZENE_PDB)
    forcefield = ForceField(BENZENE_XML)
    xyz = np.array(pdb.positions.value_in_unit(nanometers))
    modeller = Modeller(pdb.topology, Quantity(xyz - xyz.mean(axis=0) + 1.3, nanometers))
    modeller.topology.setPeriodicBoxVectors([Vec3(2.6, 0, 0), Vec3(0, 2.6, 0), Vec3(0, 0, 2.6)] * nanometers)
    topology, positions, ghosts = utils.add_ghosts(
        modeller.topology, modeller.positions, molfile=BENZENE_PDB, n=2, pdb=None, rng=np.random.default_rng(5)
    )
    return topology, positions, ghosts, forcefield


def make_sampler(box, kind, tmp_path, platform="Reference", **kwargs):
    """NCMC sampler ('sphere' or 'system') attached to a Simulation (Reference platform by default)."""
    topology, positions, ghosts, forcefield = box
    system = forcefield.createSystem(
        topology, nonbondedMethod=PME, nonbondedCutoff=1.0 * nanometers, constraints=HBonds
    )
    integrator = BAOABIntegrator(TEMPERATURE, 1.0 / picosecond, DT)
    options = dict(
        system=system,
        topology=topology,
        temperature=TEMPERATURE,
        integrator=integrator,
        resname="L01",
        nPropStepsPerPert=2,
        timeStep=DT,
        ghostFile=str(tmp_path / f"{kind}-ghosts.txt"),
        log=str(tmp_path / f"{kind}.log"),
        overwrite=True,
    )
    options.update(kwargs)
    box_vectors = topology.getPeriodicBoxVectors()
    if kind == "sphere":
        # centred on the coupled benzene (residue 0), so that deletions are possible
        xyz = np.array([positions[atom.index].value_in_unit(nanometers) for atom in topology.atoms()
                        if atom.residue.index == 0 and atom.element.symbol != "H"])
        centre = xyz.mean(axis=0) * nanometers
        sampler = samplers.NonequilibriumGCMCSphereSampler(
            sphereCentre=centre, sphereRadius=1.0 * nanometers, **options
        )
    else:
        sampler = samplers.NonequilibriumGCMCSystemSampler(boxVectors=np.array(box_vectors), **options)
    simulation = Simulation(topology, system, integrator, Platform.getPlatformByName(platform))
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE, 1)
    sampler.initialise(simulation.context, simulation, ghosts)
    return sampler, simulation


def spy_on_lambdas(sampler):
    """Record every Lambda value passed to adjustSpecificMolecule."""
    calls = []
    original = sampler.adjustSpecificMolecule

    def adjust(resid, new_lambda, ele=None, vdw=None):
        calls.append(float(new_lambda))
        return original(resid, new_lambda, ele=ele, vdw=vdw)

    sampler.adjustSpecificMolecule = adjust
    return calls


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_custom_lambda_grid_sets_protocol_length(benzene_box, tmp_path, kind):
    sampler, _ = make_sampler(benzene_box, kind, tmp_path, lambdas=LAMBDAS)
    assert sampler.n_pert_steps == len(LAMBDAS) - 1
    expected = len(LAMBDAS) * 2 * DT.value_in_unit(picoseconds)
    assert np.isclose(sampler.protocol_time.value_in_unit(picoseconds), expected)


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_custom_lambda_grid_is_followed(benzene_box, tmp_path, kind):
    # No propagation, so the protocol is a pure sequence of perturbations
    sampler, simulation = make_sampler(benzene_box, kind, tmp_path, lambdas=LAMBDAS, nPropStepsPerPert=0)
    calls = spy_on_lambdas(sampler)
    sampler.move(simulation.context, 1, force="insertion")
    assert calls[: len(LAMBDAS) - 1] == LAMBDAS[1:]
    calls.clear()
    sampler.move(simulation.context, 1, force="deletion")
    assert calls[: len(LAMBDAS) - 1] == LAMBDAS[-2::-1]


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_conflicting_nPertSteps_is_ignored_with_warning(benzene_box, tmp_path, kind, caplog):
    with caplog.at_level(logging.WARNING, logger="grandlig.samplers"):
        sampler, _ = make_sampler(benzene_box, kind, tmp_path, lambdas=LAMBDAS, nPertSteps=7)
    assert sampler.n_pert_steps == len(LAMBDAS) - 1
    assert "nPertSteps=7 ignored" in caplog.text


def run_moves(box, kind, tmp_path, seed, n_moves=3, platform="Reference"):
    """Run random NCMC moves with a seeded sampler; return everything that depends on random draws."""
    sampler, simulation = make_sampler(box, kind, tmp_path, platform=platform, seed=seed, nPertSteps=4)
    sampler.move(simulation.context, n_moves)
    tracked = sampler.tracked_variables
    works = [
        w.value_in_unit(w.unit) if hasattr(w, "unit") else w
        for w in tracked["insert_works"] + tracked["delete_works"]
    ]
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    return {
        "move_resi": [int(r) if r is not None else None for r in tracked["move_resi"]],
        "outcome": list(tracked["outcome"]),
        "n_inserts": tracked["n_inserts"],
        "acc": np.array(tracked["acceptance_probabilities"], dtype=float),
        "works": np.array(works, dtype=float),
        "positions": positions.value_in_unit(nanometers),
        "integrator_seed": sampler.integrator.getRandomNumberSeed(),
    }


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_seeded_runs_are_reproducible(benzene_box, tmp_path, kind):
    a = run_moves(benzene_box, kind, tmp_path, seed=11)
    b = run_moves(benzene_box, kind, tmp_path, seed=11)
    c = run_moves(benzene_box, kind, tmp_path, seed=12)
    for key in ("move_resi", "outcome", "n_inserts", "integrator_seed"):
        assert a[key] == b[key], key
    for key in ("acc", "works", "positions"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["integrator_seed"] != 0
    assert a["integrator_seed"] != c["integrator_seed"]
    # rejected moves restore the positions, so compare the protocol works instead
    assert not np.array_equal(a["works"], c["works"], equal_nan=True)


def test_seeded_sampler_does_not_use_global_random_state(benzene_box, tmp_path):
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    run_moves(benzene_box, "sphere", tmp_path, seed=3, platform="CPU")
    run_moves(benzene_box, "system", tmp_path, seed=3, platform="CPU")
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_unseeded_sampler_leaves_integrator_seed_alone(benzene_box, tmp_path):
    sampler, _ = make_sampler(benzene_box, "system", tmp_path)
    assert sampler.seed is None
    assert sampler.integrator.getRandomNumberSeed() == 0


def test_random_rotation_matrix_with_generator():
    a = utils.random_rotation_matrix(np.random.default_rng(1))
    b = utils.random_rotation_matrix(np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(a), 1.0)
    # the default still draws from numpy's global state
    np.random.seed(2)
    c = utils.random_rotation_matrix()
    np.random.seed(2)
    np.testing.assert_array_equal(c, utils.random_rotation_matrix())


def test_add_ghosts_with_generator():
    pdb = PDBFile(BENZENE_PDB)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.topology.setPeriodicBoxVectors([Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)] * nanometers)
    runs = [
        utils.add_ghosts(modeller.topology, modeller.positions, molfile=BENZENE_PDB, n=3, pdb=None,
                         rng=np.random.default_rng(9))
        for _ in range(2)
    ]
    x0, x1 = (np.array(r[1].value_in_unit(nanometers)) for r in runs)
    np.testing.assert_array_equal(x0, x1)
    assert runs[0][2] == runs[1][2] == [1, 2, 3]


class ExplodingIntegrator:
    """Wraps an integrator; the n-th call to step() raises, as an unstable integration would."""

    def __init__(self, integrator, fail_on_call):
        self.integrator, self.fail_on_call, self.calls = integrator, fail_on_call, 0

    def step(self, n):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated explosion")
        return self.integrator.step(n)


def assert_one_record_per_move(sampler):
    tracked = sampler.tracked_variables
    assert len(tracked["outcome"]) == tracked["n_moves"], tracked["outcome"]
    assert len(tracked["move_resi"]) == tracked["n_moves"]
    assert len(tracked["acceptance_probabilities"]) == tracked["n_moves"]
    assert len(tracked["insert_works"]) == tracked["n_inserts"]
    assert len(tracked["delete_works"]) == tracked["n_deletes"]


@pytest.mark.parametrize("kind", ["sphere", "system"])
@pytest.mark.parametrize("direction", ["insertion", "deletion"])
def test_explosion_is_recorded_once(benzene_box, tmp_path, kind, direction):
    sampler, simulation = make_sampler(benzene_box, kind, tmp_path, platform="CPU", seed=1, nPertSteps=3)
    sampler.integrator = ExplodingIntegrator(sampler.integrator, fail_on_call=3)
    sampler.move(simulation.context, 1, force=direction)
    assert sampler.tracked_variables["outcome"] == ["explosion"]
    assert sampler.tracked_variables["n_explosions"] == 1
    works = sampler.tracked_variables["insert_works" if direction == "insertion" else "delete_works"]
    assert len(works) == 1 and np.isnan(works[0])
    assert_one_record_per_move(sampler)


def test_left_sphere_is_recorded_once(benzene_box, tmp_path):
    # no propagation: the molecule cannot explode and stays where it was put
    sampler, simulation = make_sampler(
        benzene_box, "sphere", tmp_path, platform="CPU", seed=1, nPertSteps=2, nPropStepsPerPert=0
    )
    ghost = sampler.getMolStatusResids(0)[0]

    def insert_outside():
        # put the ghost's centre 1.2 nm from the centre of the 1.0 nm sphere
        xyz = np.array(sampler.positions.value_in_unit(nanometers))
        atoms, heavy = sampler.mol_atom_ids[ghost], sampler.mol_heavy_ids[ghost]
        target = np.array(sampler.sphere_centre.value_in_unit(nanometers)) + np.array([1.2, 0.0, 0.0])
        xyz[atoms] += target - xyz[heavy].mean(axis=0)
        return Quantity(xyz, nanometers), ghost

    sampler.insertRandomMolecule = insert_outside
    sampler.move(simulation.context, 1, force="insertion")
    assert sampler.tracked_variables["outcome"] == ["left_sphere"]
    assert_one_record_per_move(sampler)


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_failed_deletion_is_recorded_once(benzene_box, tmp_path, kind):
    sampler, simulation = make_sampler(benzene_box, kind, tmp_path, platform="CPU", seed=1, nPertSteps=2)
    sampler.deleteRandomMolecule = lambda: None
    sampler.move(simulation.context, 1, force="deletion")
    assert sampler.tracked_variables["outcome"] == ["no_mol2del"]
    assert_one_record_per_move(sampler)


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_random_moves_keep_records_aligned(benzene_box, tmp_path, kind):
    sampler, simulation = make_sampler(benzene_box, kind, tmp_path, platform="CPU", seed=4, nPertSteps=2)
    sampler.move(simulation.context, 6)
    assert_one_record_per_move(sampler)


def kcal(work):
    """Work entries are Quantities, or NaN floats for failed moves."""
    return work.value_in_unit(kilocalories_per_mole) if is_quantity(work) else float(work)


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_work_trace_is_off_by_default(benzene_box, tmp_path, kind):
    sampler, simulation = make_sampler(benzene_box, kind, tmp_path, platform="CPU", seed=1, nPertSteps=2)
    sampler.move(simulation.context, 2)
    assert sampler.record_work_trace is False
    assert "work_traces" not in sampler.tracked_variables


@pytest.mark.parametrize("kind", ["sphere", "system"])
def test_work_trace_sums_to_protocol_work(benzene_box, tmp_path, kind):
    sampler, simulation = make_sampler(
        benzene_box, kind, tmp_path, lambdas=LAMBDAS, seed=2, recordWorkTrace=True
    )
    sampler.move(simulation.context, 1, force="deletion")
    sampler.move(simulation.context, 1, force="insertion")
    deletion, insertion = sampler.tracked_variables["work_traces"]
    assert (deletion["move"], deletion["direction"]) == (0, "deletion")
    assert (insertion["move"], insertion["direction"]) == (1, "insertion")
    assert deletion["lambdas"] == LAMBDAS[-2::-1]
    assert insertion["lambdas"] == LAMBDAS[1:]
    for trace, works in ((deletion, "delete_works"), (insertion, "insert_works")):
        total = kcal(sampler.tracked_variables[works][-1])
        assert len(trace["work_steps"]) == len(LAMBDAS) - 1
        if np.isfinite(total):
            assert np.isclose(sum(trace["work_steps"]), total, rtol=1e-12, atol=1e-9)


def test_work_trace_is_kept_for_explosions(benzene_box, tmp_path):
    sampler, simulation = make_sampler(
        benzene_box, "system", tmp_path, platform="CPU", seed=1, nPertSteps=4, recordWorkTrace=True
    )
    sampler.integrator = ExplodingIntegrator(sampler.integrator, fail_on_call=3)
    sampler.move(simulation.context, 1, force="deletion")
    (trace,) = sampler.tracked_variables["work_traces"]
    # the initial propagation is call 1; the perturbations before calls 2 and 3 are recorded
    assert len(trace["work_steps"]) == 2
    assert sampler.tracked_variables["outcome"] == ["explosion"]
