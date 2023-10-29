#!/usr/bin/env python3
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    ToroidalFlux, MajorRadius, CurveLength, CurveCurveDistance, NonQuasiSymmetricRatio, Iotas, BoozerResidual, \
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation, Volume, CurveSurfaceDistance
from simsopt._core import load, save
from simsopt.objectives import MPIObjective, MPIOptimizable
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
from qsc import Qsc
import simsoptpp as sopp
import numpy as np
import os
import sys
import time

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    def pprint(*args, **kwargs):
        if comm.rank == 0:  # only print on rank 0
            print(*args, **kwargs)

except ImportError:
    comm = None
    size = 1
    pprint = print

def linking_number(curves):
    try:
        ln_coils = 0
        for i in range(len(curves)):
            for j in range(i+1, len(curves)):
                ln_coils += abs(sopp.ln(curves[i].gamma(), curves[j].gamma()))
    except:
        ln_coils=100
    return ln_coils

ns = int(sys.argv[1])

pprint(f"Start of boozerLS surace {ns}")

# Directory for output
OUT_DIR = f"./output_ls{ns}"
os.makedirs(OUT_DIR, exist_ok=True)
pprint("================================")

try:
    [surfaces, boozer_surfaces, ress, coils, axis, problem_config] = load(f'./ls{ns-1}/boozer_surface_0.json')
except:
    pprint("not found")
    exit(1)

ma = axis[0]
etabar = axis[1]
nfp = ma.nfp
stellsym = ma.stellsym

curves = [c.curve for c in coils]
phis = np.linspace(0, 1/nfp, 30, endpoint=False)
thetas = np.linspace(0, 1,   30, endpoint=False)

def surface_continuation(target, surface, iota0, G0, w0):
    phis = np.linspace(0, 1/nfp, 30, endpoint=False)
    thetas = np.linspace(0, 1,   30, endpoint=False)
    w = w0
    mn_list = [(2, 2), (3, 3), (4, 4)]
    idx_first = mn_list.index((surface.mpol, surface.ntor))
    mn_list = mn_list[idx_first:]
    save_boozer_surface = None
    for (mpol, ntor) in mn_list:
        snew = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        snew.least_squares_fit(surface.gamma())
        surface = snew
        for it in range(10):
            boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), target, constraint_weight=w)
            res = boozer_surface.run_code('ls', iota0, G0)
            iota0, G0 = res['iota'], res['G']
            label_err = np.abs((boozer_surface.label.J()-target)/target)
            is_inter = np.any([surface.is_self_intersecting(a) for a in np.linspace(0, 2*np.pi/nfp, 10)])
            pprint(rank, (mpol, ntor), res['success'], is_inter, label_err, (label_err > 0.001 and res['success'] and not is_inter) )

            if label_err > 0.001 and res['success'] and not is_inter:
                w*=10
            else:
                break
        if res['success'] and not is_inter and label_err <= 0.001:
            save_boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), target, constraint_weight=w)
            res = save_boozer_surface.run_code('ls', iota0, G0)
        else:
            break
    return save_boozer_surface

def run_surface_continuation(init_surfaces, init_iotaG, vol_list):
    boozer_surfaces = []
    surfaces = []
    
    for label, surface, (iota0, G0, w0) in zip(vol_list, init_surfaces, init_iotaG):
        boozer_surface = surface_continuation(label, surface, iota0, G0, w0)
        if boozer_surface is not None:
            iota0, G0 = boozer_surface.res['iota'], boozer_surface.res['G']
            boozer_surfaces.append(boozer_surface)
            surfaces.append(boozer_surface.surface)
        else:
            pprint("failed on surface solve")
            exit(1)
    return surfaces, boozer_surfaces

is_inter = [np.any([surface.is_self_intersecting(a) for a in np.linspace(0, 2*np.pi, 10)]) for surface in surfaces]
init_surfaces = surfaces
init_iotaG = [(res['iota'], res['G'], res['w']) for res in ress]

snew = SurfaceXYZTensorFourier(mpol=2, ntor=2, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
snew.least_squares_fit(surfaces[-1].gamma())
init_surfaces.append(snew)
init_iotaG.append(init_iotaG[-1])

vol_list = [2*np.pi**2 * 1. * mr**2 * np.sign(surface.volume()) for mr, surface in zip(np.linspace(0.05, 0.35, 7)[:ns], surfaces)]
surfaces, boozer_surfaces = run_surface_continuation(init_surfaces, init_iotaG, vol_list)

init_surfaces = surfaces
init_iotaG = [(bs.res['iota'], bs.res['G'], bs.constraint_weight) for bs in boozer_surfaces]
for s in surfaces:
    pprint((s.mpol, s.ntor), s.aspect_ratio(), s.major_radius(), s.volume())

nfp = ma.nfp
nc_per_hp = len(coils)//nfp//2
base_curves = [coils[i].curve for i in range(nc_per_hp)]
base_currents = [coils[i].current for i in range(nc_per_hp)]
curves = [c.curve for c in coils]

# initial weighting...
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
JBoozerResidual = MPIObjective(brs, comm, needs_splitting=True)

MIN_DIST_THRESHOLD = 0.1
KAPPA_THRESHOLD = 5.
MSC_THRESHOLD = 5.
MR_TARGET = 1.
IOTAS_TARGET     = problem_config['iota_target']
LENGTH_THRESHOLD = problem_config['clen_target']

RES_WEIGHT       = problem_config['res_weight']
LENGTH_WEIGHT    = problem_config['clen_weight']
MR_WEIGHT        = problem_config['mr_weight']
MIN_DIST_WEIGHT  = problem_config['min_dist_weight']
MIN_CS_DIST_WEIGHT  = problem_config['min_cs_dist_weight']
KAPPA_WEIGHT     = problem_config['curvature_weight']
MSC_WEIGHT       = problem_config['msc_weight']
IOTAS_WEIGHT     = problem_config['iota_weight']
ARCLENGTH_WEIGHT = problem_config['alen_weight']

base_currents[0].fix_all()

for i in range(3):
    order = [(s.mpol, s.ntor) for s in surfaces]
    if np.any([t != (4, 4) for t in order]):
        surfaces, boozer_surfaces = run_surface_continuation(init_surfaces, init_iotaG, vol_list)
        
        for surface in surfaces:
            pprint((surface.mpol, surface.ntor), surface.aspect_ratio(), surface.major_radius(), surface.volume())
        
        nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
        brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
        JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
        JBoozerResidual = MPIObjective(brs, comm, needs_splitting=True)
        # the residual weighting depends on the order of approximation
        RES_WEIGHT = 10*JnonQSRatio.J() / JBoozerResidual.J()

    mpi_surfaces = MPIOptimizable(surfaces, ["x"], comm)
    mpi_boozer_surfaces = MPIOptimizable(boozer_surfaces, ["res", "need_to_run_code"], comm)
    
    mrs = [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces]
    iotas = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
    nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
    brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
    
    mean_iota = MPIObjective(iotas, comm, needs_splitting=True)
    
    Jiotas = QuadraticPenalty(mean_iota, IOTAS_TARGET, 'identity')
    JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
    JBoozerResidual = MPIObjective(brs, comm, needs_splitting=True)
    Jmajor_radius = MPIObjective([len(mrs)*QuadraticPenalty(mr, MR_TARGET, 'identity') if idx == (len(mrs) - 1) else 0*QuadraticPenalty(mr, mr.J(), 'identity') for idx, mr in enumerate(mrs)], comm, needs_splitting=True)
    
    ls = [CurveLength(c) for c in base_curves]
    Jls = QuadraticPenalty(sum(ls), LENGTH_THRESHOLD, 'max')
    Jccdist = CurveCurveDistance(curves, MIN_DIST_THRESHOLD)
    Jcsdist = CurveSurfaceDistance(curves, mpi_surfaces[-1], MIN_DIST_THRESHOLD)
    Jcs = sum([LpCurveCurvature(c, 2, KAPPA_THRESHOLD) for c in base_curves])
    msc_list = [MeanSquaredCurvature(c) for c in base_curves]
    Jmsc = sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in msc_list)
    Jals = sum([ArclengthVariation(c) for c in base_curves])
    
    JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiotas + MR_WEIGHT * Jmajor_radius \
        + LENGTH_WEIGHT * Jls + MIN_DIST_WEIGHT * Jccdist + MIN_CS_DIST_WEIGHT * Jcsdist + KAPPA_WEIGHT * Jcs\
        + MSC_WEIGHT * Jmsc \
        + ARCLENGTH_WEIGHT * Jals
    
    # dictionary used to save the last accepted surface dofs in the line search, in case Newton's method fails
    prevs = {'sdofs': [surface.x.copy() for surface in mpi_surfaces], 'iota': [boozer_surface.res['iota'] for boozer_surface in mpi_boozer_surfaces],
             'G': [boozer_surface.res['G'] for boozer_surface in mpi_boozer_surfaces], 'J': JF.J(), 'dJ': JF.dJ().copy(), 'it': 0}
    
    def fun(dofs):
        # initialize to last accepted surface values
        for idx, surface in enumerate(mpi_surfaces):
            surface.x = prevs['sdofs'][idx]
        for idx, boozer_surface in enumerate(mpi_boozer_surfaces):
            boozer_surface.res['iota'] = prevs['iota'][idx]
            boozer_surface.res['G'] = prevs['G'][idx]
        
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()

        ln_coils = linking_number([c.curve for c in coils])
        res_success = np.all([boozer_surface.res['success'] for boozer_surface in mpi_boozer_surfaces]) 
        si_success  = np.all([np.all([not surface.is_self_intersecting(a) for a in np.linspace(0, np.pi/surface.nfp, 5)]) for surface in mpi_surfaces])
        ln_success  = ln_coils == 0 
        success = res_success and si_success and ln_success
        if not success:
            pprint(f"not success debug: all converged? {res_success}\n all non-self-intersecting? {si_success}\n linking number is zero ? {ln_success}, ln_coils {ln_coils}\n")
            J = prevs['J']
            grad = -prevs['dJ']
            for idx, boozer_surface in enumerate(mpi_boozer_surfaces):
                boozer_surface.surface.x = prevs['sdofs'][idx]
                boozer_surface.res['iota'] = prevs['iota'][idx]
                boozer_surface.res['G'] = prevs['G'][idx]
        return J, grad
    
    def callback(x):
        # since this set of coil dofs was accepted, set the backup surface dofs
        # to accepted ones in case future Newton solves fail.
        for idx, surface in enumerate(mpi_surfaces):
            prevs['sdofs'][idx] = surface.x.copy()
        for idx, boozer_surface in enumerate(mpi_boozer_surfaces):
            prevs['iota'][idx] = boozer_surface.res['iota']
            prevs['G'][idx] = boozer_surface.res['G']
        prevs['J'] = JF.J()
        prevs['dJ'] = JF.dJ().copy()

        width = 35
        outstr = f"\nIteration {prevs['it']}\n"
        outstr += f"{'J':{width}} {JF.J():.6e} \n"
        outstr += f"{'║∇J║':{width}} {np.linalg.norm(JF.dJ()):.6e} \n\n"
        outstr += f"{'nonQS ratio':{width}}" + ", ".join([f'{np.sqrt(nonqs.J()):.6e}' for nonqs in nonQSs])+ f", ({JnonQSRatio.J():.6f})" + "\n"
        outstr += f"{'Boozer Residual':{width}}" + ", ".join([f'{br.J():.6e}' for br in brs]) + f", ({RES_WEIGHT * JBoozerResidual.J():.6f})"+ "\n"
        outstr += f"{'<ι>':{width}} {mean_iota.J():.6f}"+ f" ({IOTAS_WEIGHT * Jiotas.J():.6f})" + "\n"
        outstr += f"{'ι on surfaces':{width}}" + ", ".join([f"{boozer_surface.res['iota']:.6f}" for boozer_surface in boozer_surfaces]) + "\n"
        outstr += f"{'major radius on surfaces':{width}}" + ', '.join([f'{surface.major_radius():.6f}' for surface in surfaces]) + f", ({MR_WEIGHT * Jmajor_radius.J():6f})"+ "\n"
        outstr += f"{'minor radius on surfaces':{width}}" + ', '.join([f'{surface.minor_radius():.6f}' for surface in surfaces]) + "\n"
        outstr += f"{'aspect ratio on surfaces':{width}}" + ', '.join([f'{surface.aspect_ratio():.6f}' for surface in surfaces]) + "\n"
        outstr += f"{'volume':{width}}" + ', '.join([f'{surface.volume():.6f}' for surface in surfaces]) + "\n"
        outstr += f"{'surfaces are self-intersecting':{width}}" + ', '.join([f'{surface.is_self_intersecting()}' for surface in surfaces]) + "\n"
        outstr += f"{'shortest coil to coil distance':{width}} {Jccdist.shortest_distance():.3f}" + f", ({MIN_DIST_WEIGHT * Jccdist.J():.6f})"  +"\n"
        outstr += f"{'shortest coil to surface distance':{width}} {Jcsdist.shortest_distance():.3f}" + f", ({MIN_CS_DIST_WEIGHT * Jcsdist.J():.6f})"  +"\n"
        outstr += f"{'coil lengths':{width}}" + ', '.join([f'{J.J():5.6f}' for J in ls]) + "\n"
        outstr += f"{'coil length sum':{width}} {sum(J.J() for J in ls):.3f}" + f", ({LENGTH_WEIGHT * Jls.J():.6f})" + "\n"
        outstr += f"{'max κ':{width}}" + ', '.join([f'{np.max(c.kappa()):.6f}' for c in base_curves]) + f", ({KAPPA_WEIGHT * Jcs.J():.6f})" + "\n"
        outstr += f"{'∫ κ^2 dl / ∫ dl':{width}}" + ', '.join([f'{Jmsc.J():.6f}' for Jmsc in msc_list])  + f", ({MSC_WEIGHT * Jmsc.J():.6f})"+ "\n"
        outstr += f"{'Arclength':{width}}" + ', '.join([f'{Jarc.J():.6f}' for Jarc in [ArclengthVariation(c) for c in base_curves]])  + f", ({ARCLENGTH_WEIGHT * Jals.J():.6f})"+ "\n"

        outstr += "\n\n"
        
        pprint(outstr)
        prevs['it'] += 1
    
    dofs = JF.x
    callback(dofs)
    
    pprint("""
    ################################################################################
    ### Run the optimization #######################################################
    ################################################################################
    """)
    # Number of iterations to perform:
    MAXITER = 1e2
    
    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
    xbfgs = res.x
    
    init_surfaces = [s for s in surfaces]
    init_iotaG = [(bs.res['iota'], bs.res['G'], bs.constraint_weight) for bs in boozer_surfaces]

    pprint("""
    ################################################################################
    ### Adjust the weights   #######################################################
    ################################################################################
    """)

    msc = [np.mean(c.kappa()**2 * np.linalg.norm(c.gammadash(), axis=-1))/np.mean(np.linalg.norm(c.gammadash(), axis=-1)) for c in base_curves]
    iota_err = np.abs(mean_iota.J() - IOTAS_TARGET)/np.abs(IOTAS_TARGET)
    curv_err = max(max([np.max(c.kappa()) for c in base_curves]) - KAPPA_THRESHOLD, 0)/np.abs(KAPPA_THRESHOLD)
    msc_err = max(np.max(msc) - MSC_THRESHOLD, 0)/np.abs(MSC_THRESHOLD)
    min_dist_err = max(MIN_DIST_THRESHOLD-Jccdist.shortest_distance(), 0)/np.abs(MIN_DIST_THRESHOLD)
    min_cs_dist_err = max(MIN_DIST_THRESHOLD-Jcsdist.shortest_distance(), 0)/np.abs(MIN_DIST_THRESHOLD)
    alen_err = np.max([ArclengthVariation(c).J() for c in base_curves])
    clen_err = max(sum(ls).J() - LENGTH_THRESHOLD, 0)/np.abs(LENGTH_THRESHOLD)

    pprint(f"OLD WEIGHTS iota {IOTAS_WEIGHT} curv {KAPPA_WEIGHT} min_dist {MIN_DIST_WEIGHT} min_cs_dist {MIN_CS_DIST_WEIGHT} msc {MSC_WEIGHT} alen {ARCLENGTH_WEIGHT} clen {LENGTH_WEIGHT}")
    if iota_err > 0.001:
        IOTAS_WEIGHT*=10
    if curv_err > 0.001:
        KAPPA_WEIGHT*=10
    if min_dist_err > 0.001:
        MIN_DIST_WEIGHT*=10
    if min_cs_dist_err > 0.001:
        MIN_CS_DIST_WEIGHT*=10
    if msc_err > 0.001:
        MSC_WEIGHT*=10
    if alen_err > 0.001:
        ARCLENGTH_WEIGHT*=10
    if clen_err > 0.001:
        LENGTH_WEIGHT*=10   
    pprint(f"NEW WEIGHTS iota {IOTAS_WEIGHT} curv {KAPPA_WEIGHT} min_dist {MIN_DIST_WEIGHT} min_cs_dist {MIN_CS_DIST_WEIGHT} msc {MSC_WEIGHT} alen {ARCLENGTH_WEIGHT} clen {LENGTH_WEIGHT}")

ress = []
for boozer_surface in boozer_surfaces:
    surface = boozer_surface.surface
    iota0 = boozer_surface.res['iota']
    G0 = boozer_surface.res['G']
    w = boozer_surface.constraint_weight
    res = {'surface':SurfaceXYZTensorFourier(mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym, nfp=nfp, quadpoints_phi=surface.quadpoints_phi, quadpoints_theta=surface.quadpoints_theta), 'iota':iota0, 'G':G0, 'w':w}
    res['surface'].x = surface.x
    ress.append(res)

problem_config= {'iota_weight':IOTAS_WEIGHT, 'iota_target': IOTAS_TARGET,
                 'curvature_weight':KAPPA_WEIGHT, 'curvature_target': KAPPA_THRESHOLD,
                 'min_dist_weight':MIN_DIST_WEIGHT, 'min_cs_dist_weight':MIN_CS_DIST_WEIGHT, 'min_dist_target': MIN_DIST_THRESHOLD,
                 'msc_weight':MSC_WEIGHT, 'msc_target': MSC_THRESHOLD,
                 'alen_weight':ARCLENGTH_WEIGHT, 'alen_target': 0.001,
                 'clen_weight':LENGTH_WEIGHT, 'clen_target': LENGTH_THRESHOLD,
                 'mr_target':MR_TARGET, 'mr_weight':MR_WEIGHT, 'res_weight':RES_WEIGHT,
                 'J': JF.J(), 'dJ': JF.dJ()}

save([surfaces, boozer_surfaces, ress, coils, axis, problem_config], OUT_DIR + f'boozer_surface_{rank}.json')

pprint(f"End of 2_Intermediate/boozerQA_ls.py, surface {ns}")
pprint("================================")
