//! Drop a crate onto the dock and print its height.
use box3d::*;

fn main() -> Result<(), Error> {
    println!("Box3D {:?}", box3d::version());
    let mut world = World::new(&WorldDef::default())?;

    let dock = world.create_body(&BodyDef::fixed(Pos::new(0.0, -0.5, 0.0)));
    world.create_box_shape(dock, &ShapeDef::default(), Vec3::new(20.0, 0.5, 20.0));

    let crate_body = world.create_body(&BodyDef::dynamic(Pos::new(0.0, 4.0, 0.0)));
    world.create_box_shape(crate_body, &ShapeDef::default().density(500.0), Vec3::splat(0.5));

    for i in 0..=90 {
        if i % 10 == 0 {
            let p = world.body_position(crate_body);
            println!(
                "t = {:4.2} s  height = {:.3} m  awake = {}",
                i as f32 / 60.0,
                p.y,
                world.is_body_awake(crate_body)
            );
        }
        world.step(1.0 / 60.0, 4);
    }
    Ok(())
}
