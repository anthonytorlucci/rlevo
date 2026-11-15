//! The reader the tests share: a `.rrd` file opened with rerun's own reader.

// Each test binary uses part of this.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use rerun::external::re_log_types::{EntityPath, TimelineName};
use rerun::{ChunkStore, ChunkStoreConfig, StoreKind};

/// One `.rrd` file's recording, as rerun reads it.
pub struct Recorded {
    pub application_id: String,
    pub recording_id: String,
    pub store: ChunkStore,
}

pub fn read(path: &Path) -> Recorded {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let stores = ChunkStore::from_rrd_reader(&ChunkStoreConfig::DEFAULT, &mut file)
        .expect("the file is a readable recording");
    let (id, store) = stores
        .into_iter()
        .find(|(id, _)| id.kind() == StoreKind::Recording)
        .expect("the file holds a recording store");
    Recorded {
        application_id: id.application_id().to_string(),
        recording_id: id.recording_id().to_string(),
        store,
    }
}

impl Recorded {
    /// Every entity path in the file, as text.
    pub fn entities(&self) -> BTreeSet<String> {
        self.store
            .all_entities_sorted()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// The timelines an entity's rows are on.
    pub fn timelines(&self, entity: &EntityPath) -> BTreeSet<String> {
        self.store
            .iter_physical_chunks()
            .filter(|chunk| chunk.entity_path() == entity)
            .flat_map(|chunk| {
                chunk
                    .timelines()
                    .keys()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// An entity's scalars on one timeline, as `(time, value)` in time order.
    pub fn series(&self, entity: &EntityPath, timeline: &str) -> Vec<(i64, f64)> {
        let timeline = TimelineName::try_new(timeline).expect("a timeline name");
        let component = rerun::Scalars::descriptor_scalars().component;
        let mut rows = Vec::new();
        for chunk in self.store.iter_physical_chunks() {
            if chunk.entity_path() != entity {
                continue;
            }
            let Some(column) = chunk.timelines().get(&timeline) else {
                continue;
            };
            for (time, values) in column
                .times_raw()
                .iter()
                .zip(chunk.iter_slices::<f64>(component))
            {
                rows.extend(values.iter().map(|value| (*time, *value)));
            }
        }
        rows.sort_by_key(|row| row.0);
        rows
    }

    /// The first string an entity holds under a component name.
    pub fn text(&self, entity: &EntityPath, component: &'static str) -> Option<String> {
        self.store
            .iter_physical_chunks()
            .filter(|chunk| chunk.entity_path() == entity)
            .flat_map(|chunk| {
                chunk
                    .iter_slices::<String>(component.into())
                    .flatten()
                    .map(|text| text.to_string())
                    .collect::<Vec<_>>()
            })
            .next()
    }
}

pub fn path(text: &str) -> EntityPath {
    EntityPath::from(text)
}
