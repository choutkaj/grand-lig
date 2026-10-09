"""
Description
-----------
Tests of the NCMC protocol bookkeeping in the nonequilibrium samplers: custom lambda grids,
seeding, per-move records and per-step work traces. Uses a small benzene/water box on the
Reference platform so that runs are exactly reproducible.
"""

import logging
import os

import numpy as np
import pytest
from openmm import Platform, Vec3
from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile, Simulation
from openmm.unit import kelvin, nanometers, picosecond, picoseconds
from openmmtools.integrators import BAOABIntegrator

from grandlig import samplers, utils

BENZENE_PDB = utils.get_data_file(os.path.join("tests", "Benzene.pdb"))
BENZENE_XML = utils.get_data_file(os.path.join("tests", "Benzene.xml"))
TEMPERATURE = 298 * kelvin
DT = 0.002 * picoseconds
LAMBDAS = [0.0, 0.2, 0.5, 0.75, 0.9, 1.0]


@pytest.fixture(scope="module")
def benzene_box():
    """A 2.6 nm TIP3P box with one coupled benzene (L01) and two ghost copies."""
    pdb = PDBFile(BENZENE_PDB)
    forcefield = ForceField("tip3p.xml", BENZENE_XML)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, boxSize=Vec3(2.6, 2.6, 2.6) * nanometers)
    state = np.random.get_state()
    np.random.seed(5)
    topology, positions, ghosts = utils.add_ghosts(
        modeller.topology, modeller.positions, molfile=BENZENE_PDB, n=2, pdb=None
    )
    np.random.set_state(state)
    return topology, positions, ghosts, forcefield


def make_sampler(box, kind, tmp_path, **kwargs):
    """NCMC sampler ('sphere' or 'system') attached to a Reference-platform Simulation."""
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
    simulation = Simulation(topology, system, integrator, Platform.getPlatformByName("Reference"))
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
