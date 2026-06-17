import collections
from ortools.sat.python import cp_model

def schedule(matrix, machine_capacities=[]):
    """Minimal jobshop problem with multiple machines."""
    # Data.
    jobs_data = matrix
    invalid_count = 0
    total_tasks = sum(len(job) for job in jobs_data)

    # Clean input data and count invalid entries
    cleaned_jobs_data = []
    for job in jobs_data:
        cleaned_job = []
        for task in job:
            machine, duration, pre_indexes = task
            if not isinstance(machine, int):
                machine = 0  # Default machine index if invalid
                # invalid_count += 1
            if not isinstance(duration, (int, float)):
                duration = 20  # Default duration if invalid
                invalid_count += 1
            if duration < 0:
                duration = 20
                invalid_count += 1
            cleaned_job.append((machine, duration, pre_indexes))
        cleaned_jobs_data.append(cleaned_job)

    err_rate = invalid_count / total_tasks if total_tasks > 0 else 0

    if len(machine_capacities) == 0:
        max_machine_index = 0
        for job in cleaned_jobs_data:
            for task in job:
                machine, duration, pre_indexes = task
                if machine > max_machine_index:
                    max_machine_index = machine
        machine_capacities = [1] * (max_machine_index + 1)

    machines_count = len(machine_capacities)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in cleaned_jobs_data for task in job)

    if horizon == 0:
        print("No tasks in the job data.")
        return {}, None, err_rate

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(cleaned_jobs_data):
        if len(job) == 0:
            continue
        for task_id, task in enumerate(job):
            machine, duration, pre_indexes = task
            duration = int(duration)
            suffix = f"_{job_id}_{task_id}"
            start_var = model.new_int_var(0, int(horizon), "start" + suffix)
            end_var = model.new_int_var(0, int(horizon), "end" + suffix)
            interval_var = model.new_interval_var(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # Create and add capacity constraints for each machine.
    for machine in all_machines:
        intervals = machine_to_intervals[machine]
        if machine_capacities[machine] == 1:
            model.add_no_overlap(intervals)
        else:
            demands = [1] * len(intervals)
            model.add_cumulative(intervals, demands, machine_capacities[machine])

    # Precedences inside a job and based on pre_indexes.
    for job_id, job in enumerate(cleaned_jobs_data):
        for task_id, task in enumerate(job):
            machine, duration, pre_indexes = task
            for pre_index in pre_indexes:
                # Check if the pre_index is a valid task index in the current job
                if not isinstance(pre_index, int) or pre_index < 0 or pre_index >= len(job):
                    invalid_count += 1  # Increment invalid count
                    continue  # Skip invalid pre_index
                
                # Add a constraint that ensures the pre-task must finish before the current task starts.
                model.add(
                    all_tasks[job_id, pre_index].end <= all_tasks[job_id, task_id].start
                )

    # Makespan objective.
    obj_var = model.new_int_var(0, int(horizon), "makespan")
    model.add_max_equality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(cleaned_jobs_data) if len(job) > 0],
    )
    model.minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    assigned_jobs = collections.defaultdict(list)
    makespan = -1
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        makespan = int(solver.objective_value)
        print("Solution:")
        # Create one list of assigned tasks per machine.
        for job_id, job in enumerate(cleaned_jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    (solver.value(all_tasks[job_id, task_id].start),
                     job_id,
                     task_id,
                     task[1])
                )

        # Create per machine output lines.
        output = ""
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort(key=lambda x: x[0])
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task[1]}_task_{assigned_task[2]}"
                # add spaces to output to align columns.
                sol_line_tasks += f"{name:15}"

                start = assigned_task[0]
                duration = assigned_task[3]
                sol_tmp = f"[{start},{start + duration}]"
                # add spaces to output to align columns.
                sol_line += f"{sol_tmp:15}"

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {makespan}")
        print(output)
    else:
        print("No solution found.")

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")
    return assigned_jobs, solver, err_rate, makespan