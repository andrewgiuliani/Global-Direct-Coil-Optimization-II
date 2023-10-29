#!/usr/bin/env python3
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    ToroidalFlux, MajorRadius, CurveLength, CurveCurveDistance, NonQuasiSymmetricRatio, Iotas, BoozerResidual, \
    LpCurveCurvature, MeanSquaredCurvature, ArclengthVariation, Volume, CurveXYZFourier, CurveSurfaceDistance
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

ns = int(sys.argv[1])

pprint(f"Start of boozerExact surace {ns}")

# Directory for output
OUT_DIR = f"./ex{ns}/"
os.makedirs(OUT_DIR, exist_ok=True)
pprint("================================")

try:
    [surfaces_old, boozer_surfaces_old, ress, coils_old, axis, problem_config, xbfgs, sigma] = load(f'./ls{ns}/boozer_surface_0.json')
    pprint(boozer_surfaces_old[0].x.size, coils_old[0].curve.quadpoints.size)
    
    nfp = axis[0].nfp
    stellsym = axis[0].stellsym
    nc_per_hp = len(coils_old)//nfp//2
    base_curves_old = [c.curve for c in coils_old[:nc_per_hp]] 
    base_currents_old = [c.current for c in coils_old[:nc_per_hp]] 
    
    base_curves = []
    base_currents = []
    curves = []
    surfaces = []
    boozer_surfaces =[]
    

    for c in base_curves_old:
        temp = CurveXYZFourier(c.quadpoints.size, 16)
        temp.least_squares_fit(c.gamma())
        cnew = CurveXYZFourier(16*10, 16)
        cnew.x = temp.x
        base_curves.append(cnew)
    
    base_currents = base_currents_old

    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    for s, bs in zip(surfaces_old, boozer_surfaces_old):
        surface = SurfaceXYZTensorFourier(mpol=s.mpol, ntor=s.ntor, stellsym=s.stellsym, nfp=s.nfp, quadpoints_phi=s.quadpoints_phi, quadpoints_theta=s.quadpoints_theta)
        surface.x = s.x

        boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), bs.targetlabel, constraint_weight=bs.constraint_weight)
        surfaces.append(surface)
        boozer_surfaces.append(boozer_surface)
    
    pprint(boozer_surfaces[0].x.size, coils[0].curve.quadpoints.size)

except:
    pprint("not found")
    exit(1)

ma = axis[0]
etabar = axis[1]
nfp = ma.nfp
stellsym = ma.stellsym
curves = [c.curve for c in coils]
lo_rational = np.concatenate( (nfp/np.arange(1, 15), 2*nfp/np.arange(1, 15)) )
iota_list = [res['iota'] for res in ress]

def surface_continuation(target, surface, iota0, G0, w0=1, kind='ls'):

    save_boozer_surface = None
    mn_list = [(2, 2), (3, 3), (4, 4)] if kind =='ls' else [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
    idx_first = -1
    if (surface.mpol, surface.ntor) in mn_list:
        idx_first = mn_list.index((surface.mpol, surface.ntor))
    mn_list = mn_list[idx_first:]

    for (mpol, ntor) in mn_list:
        if kind == 'exact':
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        else:
            phis = np.linspace(0, 1/nfp, 30, endpoint=False)
            thetas = np.linspace(0, 1, 30, endpoint=False)

        sold = SurfaceXYZTensorFourier(mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym, nfp=surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        sold.x = surface.x

        snew = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        snew.least_squares_fit(sold.gamma())
        surface = snew
        
        if kind == 'exact':
            boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), target)
            res = boozer_surface.run_code('exact', iota0, G0)
        else:
            w = w0
            for it in range(10):
                boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), target, constraint_weight=w)
                res = boozer_surface.run_code('ls', iota0, G0)
                iota0, G0 = res['iota'], res['G']
                label_err = np.abs((boozer_surface.label.J()-target)/target)
                is_inter = np.any([surface.is_self_intersecting(a) for a in np.linspace(0, np.pi/nfp, 10)])
                pprint(rank, (mpol, ntor), res['success'], is_inter, label_err, (label_err > 0.001 and res['success'] and not is_inter) )
                
                if label_err > 0.001 and res['success'] and not is_inter:
                    w*=10
                else:
                    break
        iota0, G0 = res['iota'], res['G']
        label_err = np.abs((boozer_surface.label.J()-target)/target)
        is_inter = np.any([surface.is_self_intersecting(a) for a in np.linspace(0, np.pi/nfp, 10)])
        pprint((mpol, ntor), res['success'], is_inter)

        if res['success'] and not is_inter:
            if kind == 'exact':
                save_boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), target)
            else:
                save_boozer_surface = BoozerSurface(BiotSavart(coils), surface, Volume(surface), target, constraint_weight=w)
            res = save_boozer_surface.run_code(kind, iota0, G0)
        else:
            break
    
    return save_boozer_surface

def run_surface_continuation(init_surfaces, init_iotaG, vol_list):
    boozer_surfaces = []
    surfaces = []
    iota_list = [iotaG[0] for iotaG in init_iotaG]
    exacts = [not np.any(np.abs((iot-lo_rational)/iot) < 0.001) for iot in iota_list]
 
    for vol, surface, res, exact in zip(vol_list, init_surfaces, init_iotaG, exacts):
        iota0, G0 = res
        boozer_surface = None
        if exact:
            boozer_surface = surface_continuation(vol, surface, iota0, G0, kind='exact')
        
        if boozer_surface is None:
            boozer_surface = surface_continuation(vol, surface, iota0, G0, kind='ls')

        if boozer_surface is not None:
            iota0, G0 = boozer_surface.res['iota'], boozer_surface.res['G']
            boozer_surfaces.append(boozer_surface)
            surfaces.append(boozer_surface.surface)
        else:
            pprint("failed on surface solve")
            exit(1)
    return surfaces, boozer_surfaces

def linking_number(curves):
    try:
        ln_coils = 0
        for i in range(len(curves)):
            for j in range(i+1, len(curves)):
                ln_coils += abs(sopp.ln(curves[i].gamma(), curves[j].gamma()))
    except:
        ln_coils=100
    return ln_coils

is_inter = [np.any([surface.is_self_intersecting(a) for a in np.linspace(0, 2*np.pi, 10)]) for surface in surfaces]
init_surfaces = surfaces
init_iotaG = [(res['iota'], res['G']) for res in ress]
vol_list = [2*np.pi**2 * 1. * mr**2 * np.sign(surface.volume()) for mr, surface in zip(np.linspace(0.05, 0.35, 7)[:ns], surfaces)]
surfaces, boozer_surfaces = run_surface_continuation(init_surfaces, init_iotaG, vol_list)

init_surfaces = surfaces
init_iotaG = [ (res['iota'], res['G']) for res in ress]
exacts = [bs.res['type'] == 'exact' for bs in boozer_surfaces]

for surface in surfaces:
    pprint((surface.mpol, surface.ntor), surface.aspect_ratio(), surface.major_radius(), surface.volume())

nfp = ma.nfp
nc_per_hp = len(coils)//nfp//2
base_curves = [coils[i].curve for i in range(nc_per_hp)]
base_currents = [coils[i].current for i in range(nc_per_hp)]
curves = [c.curve for c in coils]

# initial weighting...
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
JBoozerResidual = MPIObjective([br for br, exact in zip(brs, exacts) if not exact], comm, needs_splitting=True)

MIN_DIST_THRESHOLD = 0.1
KAPPA_THRESHOLD = 5.
MSC_THRESHOLD = 5.
MR_TARGET = 1.
IOTAS_TARGET     = problem_config['iota_target']
LENGTH_THRESHOLD = problem_config['clen_target']

RES_WEIGHT       = JnonQSRatio.J()/JBoozerResidual.J() if JBoozerResidual.n > 0. else 0.
LENGTH_WEIGHT    = problem_config['clen_weight']
MR_WEIGHT        = problem_config['mr_weight']
MIN_DIST_WEIGHT  = problem_config['min_dist_weight']
MIN_CS_DIST_WEIGHT  = problem_config['min_cs_dist_weight']
KAPPA_WEIGHT     = problem_config['curvature_weight']
MSC_WEIGHT       = problem_config['msc_weight']
IOTAS_WEIGHT     = problem_config['iota_weight']
ARCLENGTH_WEIGHT = problem_config['alen_weight']

base_currents[0].fix_all()

for i in range(4):
    order = [(s.mpol, s.ntor) for s in surfaces]
    iota_list = [bs.res['iota'] for bs in boozer_surfaces]
    close_to_lor = [not np.any(np.abs((iot-lo_rational)/iot) < 0.001) for iot in iota_list]
    ls_bool = [bs.res['type'] == 'ls' for bs in boozer_surfaces]
    
    # if any of the surfaces are not at the max order, or if 
    # any surfaces are ls when not close to a low-order rational
    if np.any([t != ((10, 10) if exact else (4, 4) ) for t, exact in zip(order, exacts)]) \
            or np.any([k1 != k2 for k1, k2 in zip(close_to_lor, ls_bool)]):
        surfaces, boozer_surfaces = run_surface_continuation(init_surfaces, init_iotaG, vol_list)
        exacts = [bs.res['type'] == 'exact' for bs in boozer_surfaces]
        for surface in surfaces:
            pprint((surface.mpol, surface.ntor), surface.aspect_ratio(), surface.major_radius(), surface.volume())

        nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
        brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
        JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
        JBoozerResidual = MPIObjective([br for br, exact in zip(brs, exacts) if not exact], comm, needs_splitting=True)

        # the residual weighting depends on the order of approximation
        RES_WEIGHT = JnonQSRatio.J()/JBoozerResidual.J() if JBoozerResidual.n > 0. else 0.


    mpi_surfaces = MPIOptimizable(surfaces, ["x"], comm)
    mpi_boozer_surfaces = MPIOptimizable(boozer_surfaces, ["res", "need_to_run_code"], comm)
    
    mrs = [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces]
    iotas = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
    nonQSs = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
    brs = [BoozerResidual(boozer_surface, BiotSavart(coils)) for boozer_surface in boozer_surfaces]
    
    JBoozerResidual = MPIObjective([br for br, exact in zip(brs, exacts) if not exact], comm, needs_splitting=True)
    mean_iota = MPIObjective(iotas, comm, needs_splitting=True)
    Jiotas = QuadraticPenalty(mean_iota, IOTAS_TARGET, 'identity')
    JnonQSRatio = MPIObjective(nonQSs, comm, needs_splitting=True)
    Jmajor_radius = MPIObjective([len(mrs)*QuadraticPenalty(mr, MR_TARGET, 'identity') if idx == len(mrs)-1 else 0*QuadraticPenalty(mr, mr.J(), 'identity') for idx, mr in enumerate(mrs)], comm, needs_splitting=True)
    
    ls = [CurveLength(c) for c in base_curves]
    Jls = QuadraticPenalty(sum(ls), LENGTH_THRESHOLD, 'max')
    Jccdist = CurveCurveDistance(curves, MIN_DIST_THRESHOLD)
    #Jccdist = CurveCurveDistance(curves, MIN_DIST_THRESHOLD, num_basecurves=len(base_curves))
    Jcsdist = CurveSurfaceDistance(curves, mpi_surfaces[-1], MIN_DIST_THRESHOLD)
    Jcs = sum([LpCurveCurvature(c, 2, KAPPA_THRESHOLD) for c in base_curves])
    msc_list = [MeanSquaredCurvature(c) for c in base_curves]
    Jmsc = sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in msc_list)
    Jals = sum([ArclengthVariation(c) for c in base_curves])
    
    JF = JnonQSRatio + IOTAS_WEIGHT * Jiotas + MR_WEIGHT * Jmajor_radius \
        + LENGTH_WEIGHT * Jls + MIN_DIST_WEIGHT * Jccdist + MIN_CS_DIST_WEIGHT * Jcsdist + KAPPA_WEIGHT * Jcs\
        + MSC_WEIGHT * Jmsc \
        + ARCLENGTH_WEIGHT * Jals
    if JBoozerResidual.n > 0:
        JF += RES_WEIGHT * JBoozerResidual

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
        
        alldofs = MPI.COMM_WORLD.allgather(dofs)
        assert np.all(np.norm(alldofs[0]-d) == 0 for d in alldofs)
     
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
        outstr += f"{'surface type':{width}}" + ", ".join([f'{"exact" if kind else "ls"}' for kind in exacts]) +  "\n"
        outstr += f"{'nonQS ratio':{width}}" + ", ".join([f'{np.sqrt(nonqs.J()):.6e}' for nonqs in nonQSs])+ f", ({JnonQSRatio.J():.6f})" + "\n"
        outstr += f"{'Boozer Residual':{width}}" + ", ".join([f'{br.J():.6e}' if not exact else f"{0.:.6e}" for br, exact in zip(brs, exacts)]) + f", ({RES_WEIGHT * JBoozerResidual.J() if JBoozerResidual.n > 0 else 0. :.6f})"+ "\n"
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
    
    if False in exacts:
        # Number of iterations to perform:
        MAXITER = 1e2 if i == 0 else 1e3
    else:
        MAXITER = 1e2 if i == 0 else 5e3

    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
    xbfgs = res.x
    
    pprint(res.message)
    
    init_surfaces = [s for s in surfaces]
    init_iotaG = [(bs.res['iota'], bs.res['G']) for bs in boozer_surfaces]

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
    
    if iota_err <= 0.001 and curv_err <= 0.001 and min_dist_err <=0.001 and msc_err <= 0.001 and alen_err <= 0.001 and clen_err <= 0.001 and np.linalg.norm(JF.dJ(), ord=np.inf) <= 1e-7:
        pprint("OPTIMIZATION PROBLEM SOLVED!")
        break

    pprint(f"OLD WEIGHTS iota {IOTAS_WEIGHT} curv {KAPPA_WEIGHT} min_dist {MIN_DIST_WEIGHT} min_cs_dist {MIN_CS_DIST_WEIGHT} msc {MSC_WEIGHT} alen {ARCLENGTH_WEIGHT} clen {LENGTH_WEIGHT}")
    if iota_err > 0.001:
        IOTAS_WEIGHT*=10
    if curv_err > 0.001:
        KAPPA_WEIGHT*=10
    if min_dist_err > 0.001:
        MIN_DIST_WEIGHT*=10
    if msc_err > 0.001:
        MSC_WEIGHT*=10
    if alen_err > 0.001:
        ARCLENGTH_WEIGHT*=10
    if clen_err > 0.001:
        LENGTH_WEIGHT*=10  
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

save([surfaces, boozer_surfaces, ress, coils, axis, problem_config, xbfgs, sigma], OUT_DIR + f'boozer_surface_{rank}.json')

pprint("End of 2_Intermediate/boozerQA_ex.py")
pprint("================================")
