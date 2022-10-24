use rand::seq::SliceRandom;
use rand::Rng;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

#[derive(Debug)]
struct Arguments {
    in_file: String,
    p0: u32,
}

fn parse_args() -> Arguments {
    let args: Vec<String> = env::args().skip(1).collect();

    Arguments {
        in_file: args[0].clone(),
        p0: args[1].clone().parse().unwrap(),
    }
}
fn parse_graph(f: File) -> (Vec<(u32, u32)>, Vec<Vec<u32>>) {
    let mut reader = BufReader::new(f);
    let mut first_line: String = "".to_string();
    let _i = reader.read_line(&mut first_line);
    first_line = first_line.strip_suffix("\n").unwrap().to_owned();
    let split_first = first_line
        .split(" ")
        .map(|x| x.parse::<u32>().unwrap())
        .collect::<Vec<_>>();
    println!("{:?}", split_first);
    println!("{}", &first_line);
    let mut adj_list: Vec<Vec<u32>> = Vec::<Vec<u32>>::with_capacity(split_first[0] as usize);
    let mut edg_list: Vec<(u32, u32)> = vec![];
    for line_ in reader.lines().enumerate() {
        let line = line_.1.unwrap();
        let line: Vec<u32> = line.split(" ").map(|x| x.parse::<u32>().unwrap()).collect();

        let idx = line_.0;
        if line.len() > 0 {
            adj_list.push(vec![]);
            for numb in &line {
                adj_list[idx].push(*numb);
                if *numb > idx as u32 {
                    edg_list.push((idx as u32, *numb));
                }
            }
        }
        // println!("{:?}", line);
        // println!("{:?}", idx);
    }
    // println!("{:?}", adj_list);
    // println!("{:?}", edg_list);
    (edg_list, adj_list)
}

#[derive(Clone, Debug)]
struct Individual {
    chromosome: Vec<u32>,
    fitness: u32,
    lockdown_edgelist: Vec<(u32, u32)>,
    lockdown_adjlist: Vec<Vec<u32>>,
}

impl Individual {
    fn new() -> Self {
        todo!()
    }
}
struct Population {
    pop: Vec<Individual>,
}

impl Population {
    fn init_pop(
        percent_lock: f32,
        pop_size: i32,
        edg_list: &Vec<(u32, u32)>,
        adj_list: &Vec<Vec<u32>>,
    ) -> Self {
        let edg_count = edg_list.len();
        let mut pop_vec: Vec<Individual> = Vec::with_capacity(pop_size as usize);
        let num_remove = (percent_lock * edg_count as f32) as usize;
        let mut init_list = vec![0; edg_count];
        for (i, v) in init_list.iter_mut().enumerate() {
            *v = i as u32;
        }
        let mut rng = &mut rand::thread_rng();
        for _ in 0..pop_size {
            let mut new_list: Vec<u32> = vec![1; edg_count];
            let rindices: Vec<u32> = init_list
                .choose_multiple(&mut rng, num_remove)
                .cloned()
                .collect();
            for j in &rindices {
                new_list[*j as usize] = 0;
            }
            let (ladjlist, ledgelist) = get_lockdown_graphs(adj_list, edg_list, &new_list);
            let new_ind = Individual {
                chromosome: new_list,
                fitness: 0,
                lockdown_edgelist: ledgelist,
                lockdown_adjlist: ladjlist,
            };
            pop_vec.push(new_ind);
        }
        let newpop = Population { pop: pop_vec };
        newpop
    }
}

fn get_lockdown_graphs(
    adj_list: &Vec<Vec<u32>>,
    edg_list: &Vec<(u32, u32)>,
    remove_list: &Vec<u32>,
) -> (Vec<Vec<u32>>, Vec<(u32, u32)>) {
    let mut lock_adj: Vec<Vec<u32>> = adj_list.clone();
    let mut lock_edg: Vec<(u32, u32)> = edg_list.clone();
    let mut removeylist: Vec<usize> = vec![];
    for (idx, edge) in remove_list.iter().enumerate() {
        if *edge == 0 {
            let first = edg_list[idx].0;
            let second = edg_list[idx].1;
            if let Some(pos) = lock_adj[first as usize].iter().position(|x| *x == second) {
                lock_adj[first as usize].remove(pos);
            }
            if let Some(pos) = lock_adj[second as usize].iter().position(|x| *x == first) {
                lock_adj[second as usize].remove(pos);
            }
            removeylist.push(idx);
        }
    }
    for x in removeylist.iter().rev() {
        lock_edg.remove(*x);
    }
    (lock_adj, lock_edg)
}

pub fn set_colour(value: u32, node_num: usize) -> [u32; node_num] {
    let clr: [u32; VERTICES] = [value; VERTICES];
    clr
}

pub fn infected(n: u32, alpha: f64) -> u32 {
    let mut rng = rand::thread_rng();
    let beta: f64 = 1.0 - (f64::from(n) * (1.0 - alpha).ln()).exp();
    let x = rng.gen::<f64>();
    if x < beta {
        return 1;
    } else {
        return 0;
    }
}

pub fn SIR(&self, p0: usize, alpha: f64, node_num: usize) -> (u32, u32, u32) {
    if p0 >= node_num {
        return (0, 0, 0);
    }
    let mut max = 0;
    let mut len = 0;
    let mut ttl = 0;
    let mut nin: Vec<u32> = vec![0; node_num]; //infected neibours counters
    let mut clr: Vec<u32> = vec![0; node_num]; // set population to susceptible
    clr[p0] = 1; //infect patient zero
    let mut numb_inf = 1; // initialize to one person currently infected
    while numb_inf > 0 {
        for i in 0..VERTICES {
            nin[i] = 0; //zero the number of infected neighbors buffer
        }
        for i in 0..VERTICES {
            if clr[i] == 1 {
                //found infected individual
                for j in 0..self.adj_list[i].len() {
                    nin[self.adj_list[i][j]] += 1; //record exposure
                }
            }
        }
        //check for transmission
        for i in 0..VERTICES {
            if clr[i] == 0 && nin[i] > 0 {
                clr[i] = 3 * Unweighted::infected(nin[i], alpha);
            }
        }
        if numb_inf > max {
            max = numb_inf;
        }
        ttl += numb_inf;
        numb_inf = 0;
        for i in 0..VERTICES {
            match clr[i] {
                0 => (),         //susceptible, do nothing
                1 => clr[i] = 2, //infected, move to removed
                2 => (),         //removed, do nothing
                3 => {
                    //newly infected
                    clr[i] = 1;
                    numb_inf += 1;
                }
                _ => (),
            }
        }
        len += 1; //record time step
    }
    (max, len, ttl)
}

fn main() {
    let args = parse_args();
    println!("{}", args.p0);
    let f = File::open(&args.in_file).unwrap();

    let (elist, alist) = parse_graph(f);
    let pop = Population::init_pop(0.5, 201, &elist, &alist);

    println!("{:?}", pop.pop[4]);
    println!("{:?}", alist);
    // println!("{:?}", elist);
}
