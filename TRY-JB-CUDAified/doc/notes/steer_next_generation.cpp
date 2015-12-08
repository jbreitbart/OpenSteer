/**
 * In the first step the steering behavior functions are extremely simple.
 * They don't even use parameters to influence the magnitude of the steering
 * vector they produce.
 * 
 * In a second step their results are used and scaled or used with different
 * interpolation schemes to interpolate between them and the last position/heading
 * of the agent.
 * 
 * Separate functionality is needed to produce the sequence of data used in the
 * functions below. Functionality that might take agent transformations and
 * agent perception into account.
 * Each of the functions below needs information about the agent it is working on.
 * Therefore it might be advisable to introduce an agent data structure which holds
 * most of the data structures needed.
 * 
 * Different steering behaviors might need different data. Is this a hint that inheritance
 * is needed, that a property model is needed, that an agent data structure should just
 * reference data residing in homogeneous data blocks?
 * 
 * Could this be hidden behind a library that allows the user to set different agent properties?
 * (I don't see how to put these into a data structure at runtime without a
 * property system which increases the size and complexity of the data of an agent.
 * 
 * Add weighting template policies to have influence what is done.
 *
 * Ok, I am now sure that it isn't possible to write a library that completely hides
 * the way steering is conducted. It might be possible to develop a special steering language
 * that is feed into a library that translates it (similar to Cg or shading languages),
 * but even than it needs to be fed with data (which can be done like it is done with
 * OpenGL shading language). Is this something to try? Is there parallelization potential?
 * The problem remains that the most costly operation seems to be the
 * neighbourhood managements which shouldn't be done in a  scripting language - or should it?
 * No. But the neighbourhood management can be hidden from the user of the system.
 *
 * Problem remaining: how to represent obstacles and perhaps even level geometry?
 * Is a duplication of level geometry needed like for the physics system?
 *
 * Let the agents change themselfes with automatic updates of the neighbourhood structure
 * or shouldn't the neighbourhood structure contain references to agents? Well, it needs
 * to reference them to allow finding of nearby agents but it would only take the position
 * and perhaps other data of an agent to optimize their spatial organization but not their
 * querying (querying for agent data). Therefore agents shouldn't update their internal
 * state while the simulation is running and other agents might want to read their last state.
 *
 * Two solutions:
 * 1. the spatial data structure always references a copy of the last state.
 *  After the simulation step the agent data gets copied and the spatial data structure
 *  references the copy now
 *  (this could be done by putting vector indices into the spatial data structure and
 *  copying new agent data into a vector/array that is intended to hold the
 *  agent data of the last step. Another way would be to iterate the spatial data structure
 *  and change pointers stored in it -> looks costly).
 * 2. Agents don't change their data but "just" emit a change-command/transaction.
 *  In the update step this change is used to update the agents. Think of a
 *  command-interpreter pattern or a transaction database or something like that.
 */



// Needs positions
// Needs its agent position and possibly its heading?
void separate( vector3& result, vector3 const& position, iter start_iter, iter end_iter );


// Needs direction (and also distance?)
// Needs its agent heading
void align( vector3& result, matrix4x4 const& heading, iter start_iter, iter end_iter );


// Needs positions
// Needs its agents position
void cohesion( vector3& result, vector3 const& position, iter start_iter, iter end_iter );










