"""Q16 fix for zenoh-bridge-ros2dds 1.9.0 (B3 45): a removed route must not keep its matching listener.

RoutePublisher (and RouteServiceCli, same pattern) registered the matching listener with .background(); its callback
holds the zenoh publisher (querier), so the listener outlives the route and, at every later matching change, creates
one more DDS Reader that routes each message once more. The patch keeps the listener in the route, clears an "alive"
flag and undeclares the listener in Drop (before deleting the DDS entities), and makes the callback do nothing once
the flag is cleared (a callback already running finishes first: Drop waits for the lock).
  python3 patch_route_lifecycle.py SRC_DIR   (the zenoh-plugin-ros2dds-1.9.0 tree; edits it in place)"""

import pathlib
import sys

src = pathlib.Path(sys.argv[1]) / "zenoh-plugin-ros2dds/src"

GUARD = """                    let guard = alive.lock().unwrap_or_else(|e| e.into_inner());
                    if !*guard {
                        tracing::debug!("{route_id}: route removed, matching change ignored");
                        return;
                    }
"""

EDITS = {
    "route_publisher.rs": [
        ("    sync::{atomic::Ordering, Arc},\n", "    sync::{atomic::Ordering, Arc, Mutex},\n"),
        ("    liveliness::LivelinessToken,\n    qos::{CongestionControl, Priority, Reliability},\n",
         "    liveliness::LivelinessToken,\n    matching::MatchingListener,\n    qos::{CongestionControl, Priority, Reliability},\n"),
        ("    // the list of nodes served by this route\n    local_nodes: HashSet<String>,\n}\n",
         "    // the list of nodes served by this route\n    local_nodes: HashSet<String>,\n"
         "    // the listener activating/deactivating the DDS Reader on matching changes; undeclared when the route is dropped\n"
         "    #[serde(skip)]\n    matching_listener: Option<MatchingListener<()>>,\n"
         "    // cleared when the route is dropped, so that a late matching callback cannot create a DDS Reader any more\n"
         "    #[serde(skip)]\n    alive: Arc<Mutex<bool>>,\n}\n"),
        ("impl Drop for RoutePublisher {\n    fn drop(&mut self) {\n        self.deactivate_dds_reader();\n    }\n}\n",
         "impl Drop for RoutePublisher {\n    fn drop(&mut self) {\n"
         "        // Stop the matching listener before deleting the DDS Reader: its callback holds the zenoh publisher, so\n"
         "        // left in the background it outlives the route and, at the next matching change, creates one more DDS\n"
         "        // Reader that routes every message once more (one more copy per removed route).\n"
         "        *self.alive.lock().unwrap_or_else(|e| e.into_inner()) = false;\n"
         "        if let Some(listener) = self.matching_listener.take() {\n"
         "            if let Err(e) = listener.undeclare().wait() {\n"
         "                tracing::warn!(\"{self}: failed to undeclare the matching listener: {e}\");\n"
         "            }\n        }\n"
         "        self.deactivate_dds_reader();\n    }\n}\n"),
        ("        let dds_reader: Arc<AtomicDDSEntity> = Arc::new(DDS_ENTITY_NULL.into());\n\n"
         "        publisher\n            .matching_listener()\n            .callback({\n",
         "        let dds_reader: Arc<AtomicDDSEntity> = Arc::new(DDS_ENTITY_NULL.into());\n"
         "        let alive = Arc::new(Mutex::new(true));\n\n"
         "        let matching_listener = publisher\n            .matching_listener()\n            .callback({\n"
         "                let alive = alive.clone();\n"),
        ("                move |status| {\n                    tracing::debug!(\"{route_id} MatchingStatus changed: {status:?}\");\n"
         "                    if status.matching() {\n",
         "                move |status| {\n                    tracing::debug!(\"{route_id} MatchingStatus changed: {status:?}\");\n"
         + GUARD + "                    if status.matching() {\n"),
        ("            .background()\n            .await\n"
         "            .map_err(|e| format!(\"Failed to listen of matching status changes: {e}\",))?;\n",
         "            .await\n"
         "            .map_err(|e| format!(\"Failed to listen of matching status changes: {e}\",))?;\n"),
        ("            remote_routes: HashSet::new(),\n            local_nodes: HashSet::new(),\n        })\n",
         "            remote_routes: HashSet::new(),\n            local_nodes: HashSet::new(),\n"
         "            matching_listener: Some(matching_listener),\n            alive,\n        })\n"),
    ],
    "route_service_cli.rs": [
        ("        atomic::{AtomicBool, Ordering},\n        Arc,\n    },\n",
         "        atomic::{AtomicBool, Ordering},\n        Arc, Mutex,\n    },\n"),
        ("    liveliness::LivelinessToken,\n    query::{Querier, Reply},\n",
         "    liveliness::LivelinessToken,\n    matching::MatchingListener,\n    query::{Querier, Reply},\n"),
        ("    // the list of nodes served by this route\n    local_nodes: HashSet<String>,\n}\n",
         "    // the list of nodes served by this route\n    local_nodes: HashSet<String>,\n"
         "    // the listener activating/deactivating the DDS Reader/Writer on matching changes; undeclared on drop\n"
         "    #[serde(skip)]\n    matching_listener: Option<MatchingListener<()>>,\n"
         "    // cleared when the route is dropped, so that a late matching callback cannot activate the route any more\n"
         "    #[serde(skip)]\n    alive: Arc<Mutex<bool>>,\n}\n"),
        ("impl Drop for RouteServiceCli {\n    fn drop(&mut self) {\n        self.deactivate();\n    }\n}\n",
         "impl Drop for RouteServiceCli {\n    fn drop(&mut self) {\n"
         "        // As for RoutePublisher: stop the matching listener (its callback holds the querier) before deactivating,\n"
         "        // else it outlives the route and re-creates a DDS Reader that routes every request once more.\n"
         "        *self.alive.lock().unwrap_or_else(|e| e.into_inner()) = false;\n"
         "        if let Some(listener) = self.matching_listener.take() {\n"
         "            if let Err(e) = listener.undeclare().wait() {\n"
         "                tracing::warn!(\"{self}: failed to undeclare the matching listener: {e}\");\n"
         "            }\n        }\n"
         "        self.deactivate();\n    }\n}\n"),
        ("        let req_reader: Arc<AtomicDDSEntity> = Arc::new(DDS_ENTITY_NULL.into());\n\n"
         "        zenoh_querier\n            .matching_listener()\n            .callback({\n",
         "        let req_reader: Arc<AtomicDDSEntity> = Arc::new(DDS_ENTITY_NULL.into());\n"
         "        let alive = Arc::new(Mutex::new(true));\n\n"
         "        let matching_listener = zenoh_querier\n            .matching_listener()\n            .callback({\n"
         "                let alive = alive.clone();\n"),
        ("                move |status| {\n                        tracing::debug!(\"{route_id} MatchingStatus changed: {status:?}\");\n"
         "                        if status.matching() {\n",
         "                move |status| {\n                        tracing::debug!(\"{route_id} MatchingStatus changed: {status:?}\");\n"
         + GUARD.replace("\n                    ", "\n                        ").replace("                    let guard", "                        let guard")
         + "                        if status.matching() {\n"),
        ("            .background()\n            .await\n"
         "            .map_err(|e| format!(\"Route Service Client (ROS:{ros2_name} <-> Zenoh:{zenoh_key_expr}): failed to listen of matching status changes: {e}\",))?;\n",
         "            .await\n"
         "            .map_err(|e| format!(\"Route Service Client (ROS:{ros2_name} <-> Zenoh:{zenoh_key_expr}): failed to listen of matching status changes: {e}\",))?;\n"),
        ("            remote_routes: HashSet::new(),\n            local_nodes: HashSet::new(),\n        })\n",
         "            remote_routes: HashSet::new(),\n            local_nodes: HashSet::new(),\n"
         "            matching_listener: Some(matching_listener),\n            alive,\n        })\n"),
    ],
}

for name, edits in EDITS.items():
    f = src / name
    t = f.read_text()
    for old, new in edits:
        n = t.count(old)
        assert n == 1, (name, n, old[:80])
        t = t.replace(old, new)
    f.write_text(t)
    print(f"patched {f} ({len(edits)} edits)")
