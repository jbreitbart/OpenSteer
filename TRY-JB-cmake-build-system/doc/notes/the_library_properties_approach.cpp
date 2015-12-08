/// Ok, maybe I am missing the point with this, but I really don't see that much a problem.

/// This way our agent will only be extented by a list of behavior-function-spointers (as cheap
/// as you can get here ... you have to store this information)
// bknafla: I like this :-) Though: how to feed it with data that varies from frame to frame 
//          (look at first_try_at_something_even_more_cool_and_fancy below)
// jens: You don't feed information, the behavior takes everything it needs (from the sensor).
//       Arguing about this may sound ridiculous, but it isn't. Actually this is a different mindset.
//       The Agent has no control but completly surrenders to its behaviours. A sensor example is below.
struct behavior_I_dont_move {
	Vec3 operator() (agent_sensor &sensor) {
		return sensor.where_am_I();
	}
}

/// create different behavior by just creating structs like the one above

class agent {
	agent_sensor sensor_; // providing the access to our global world model and maybe internal data of the agent
	
	typedef Vec3 (behavior_func*) (AgendSensor&);
	std::vector<behavior_func> behavior;
	// yes I know ... pointers are evil ... use functors if you want or
	// use inheritance and put complete objects in it but this will not be cheap ... but offers
	// you to store agent specific data inside a behavior (hey, this would a cool model to write
	// code where at the end everyone would be surprised by what is happeing ... someone should try this!)
}

agent::agent()
{
#ifdef cheap_and_easy_to_create
	// fill the agent with 
	behavior.push_back (I_like_my_neighbours);
	behavior.push_back (I_dont_want_to_be_to_close_to_the_smelly_guys);
#endif
	
#ifdef cool_looking
	behavior = I_like_my_neighbours + I_dont_want_to_be_to_close_to_the_smelly_guys;
	// well, this would need some operator overloading + creates temporaries
	// ... but only when creating the agent ... will this happen often?
#endif

#ifdef cool_looking_and_fancy
	behavior = I_like_my_neighbours*2 + I_dont_want_to_be_to_close_to_the_smelly_guys*3;
	// not compatible with the design above but in this way you could even specify the
	// importance of the behavior
#endif

// bknafla (see below)
#ifdef first_try_at_something_even_more_cool_and_fancy
	// Well, I don't see how to do this easily but I want the capability 
	// (well, this capability is needed in some Plugis)
	behavior = normalize ( choose_behavior( zero == avoid_obstacles, avoid_obstacles * 6 + flee * 2, flee  ) * 1.5f + stay_on_path + wander_around );
	// It is hard to structure this (see next point).
	// Things I dislike: multiple calculations of avoid_obstacle.
	// Things that make this approach currently hard to use: how to feed data into the behavior?
	// Like the list of obstacles needed for avoid_obstacle or the path for stay_on_path.
	// One way to handle this would be to put references to the needed data into the behavior 
	// classes before adding them. However what to do if an agent needs to get access to different 
	// data from frame to frame (because he left a partition)?
	// It would be nice if it would be possible to fill the agent with behaviors and then provide a 
	// configuration how to connect these behaviors. This configuration would also have a part which 
	// specifies which data is needed and how to feed it into the behaviors.
	
	// Jens:
	// Is normalize just supposed to normalize the vector at the end of the calculation?
	// If yes, why is this part of the behavior of the agent? I think this is more of a generic design decision
	// than really part of the behavior modelling. Or is there a reason while some agent should be normalized
	// and another agent not (within the same application)?
	
	// bknafla: 
	// good question. My reasoning is that being able to scale a steering vector (avoid_stinking_guy * 5) 
	// also means that you want to normalize a steering vector for example before comparing it to a vector of
	// another combination of behaviors.

	// Jens:
	// Ah, ok, I see. Not sure if this is something that is really needed here, but I see your point. :-)
	// But I have to agree that it should be in a generic approach if it is possible to scale a vector.
	
	// bknafla: 
	// This sparks a thought - it might be (everything that starts with "it might be" should be 
	// regarded as premature design with no real application ;-) ), it might be useful to be able 
	// to feed the resulting steering vector of a behavior into another behavior (well, normalize
	// or choose are of this type) to give a hint what the behavior might want to do.

	// Jens:
	// I don't think choosing and normalizing are real behavior. A agent can not just "live" with the behavior normalize.
	// Those are just constructs which help you at expressing yourself.
	// And again I don't think that behaviors should be
	// be feed with anything. :-P They should not depend on the results of one another ... and if you really want to do this
	// than manually call it inside the behavior. ,-)
	// "We have to draw a line somewhere!"

	// bknafla: 
	// Another thought: what about returning not just one but two or a list of steering behaviors? An obstacle
	// avoidance behavior could return two vectors for the two ways around an obstacle...

	// Jens:
	// Ok, but who should than choose which of the vectors is the one to go?
	
	// Jens:
	// To the multiple calculations. I don't see a good solution to this problem atm, but if this may really hurt
	// performance badly we could think about caching the results. We may determine if the data is needed again
	// when assigning the behavior. But this is no real solution to the problem ...
	// As said above a behaviour is not fed with anything. How this can be done by using the sensors see below.
	// How to connect behaviors in a more complex way is not easy to come by. Possible ideas:
	// 1.)
	//  Do not allow it. Force behavior models, who need this to define a behavior which does implements
	//  choose_behavior( zero == avoid_obstacles, avoid_obstacles * 6 + flee * 2, flee  )
	// 2.)
	//  Force people to use strange constructs like the one below. But this has some anoying limits
	//  e.g. you have to pass functions. And how to specify the function type? Adding templates to this
	//  will not make it easier. :-) + we are looses some low level parallelism here.
	// 3.)
	//  Generate some tree like datastructure to symbolize our calculation. Complicated and
	//  make the agent bigger, but the "cleanest" approach.
	//  As a performance sidenote: Not every agent needs to store a behavior. We may generate one for every
	//  "type" of agent and than just put a reference to the behavior inside the agents. And since I don't
	//  think the behavior tree should be changed at simulation time we will not loose any parallelism.
	
	// bknafla:
	// This is something I am also thinking about - the combination of steering behaviors looks a lot like a 
	// data-flow graph. However I am starting to get the feeling that such a solution would be too general
	// purpose and therefore not performant enough... We should analyze the different plugins to see the real
	// needs of combining behaviors.

	// Jens:
	// The answer to this question is simple, but you will not like it: What do you want?
	// 1.)
	//  A solution for opensteer? Than choose option 1 or 2 I proposed above. Performacewise this should be fine.
	// 2.)
	//  A generic library approach designers could use? Then the tree is needed. But such a generic approach
	//  can not keep with the performance of the hardcoded opensteer. We should be happy if we manage
	//  to stay within the same "league" as opensteer.
#endif
}

// strange construct
struct real_choose_behavior {
	Vec3 operator(function *X, function *Y, behavior_func F1, behavior_func F2, agent_sensor sensor_) {
	// bknafla:
	// Low-level platform-specific language constructs exist that can perform the semantic equaivalent of the code
	// below in a very efficient manner. Typicalle they start with the keyword "select" - however I don't really
	// want to write such platform-dependent low-level code (but it could be wrapped...).

	// Jens:
	// No idea what you are talking about and google returns a lot of "false-positive" results. :-)
		if (X() == Y()) {
			return F1(sensor_);
		}
		return F2(sensor_);
}

behavior_func choose_behavior(function *X, function *Y, behavior_func F1, behavior_func F2, agent_sensor sensor_) {
	return boost::bind(real_choose_behavior, _1, _2, _3, _4)(X, Y, F1, F2);
	// boost::bind is a more generic implementation of std::bind1st, std::bind2nd; see http://www.boost.org/libs/bind/bind.html
	// or maybe http://www.boost.org/doc/html/lambda.html offers a better solution to this
}


Vec3  agent::steer()
{
	Vec3 where_I_want_to_be_next = sensor_.where_am_I();
	for each ( behavior_func in behavior) {
		where_I_want_to_be_next += behavior_func(sensor_);
	} 
	// hey, a for each, this means we can execute the function calls in any order, even in parallel
	
	// bknafla: 
	// This is a great idea if combining behaviors could be done by adding them - 
	// a parallel reduction. Perhaps this is even possible if something like "choose_behavior" or 
	// "normalize" (see above) is needed - but I am not quite sure about this yet.
	// If every agent would be called in parallel we have to experiment if this nested
	//   is working performantly.
	// Another experiment I would like to try is that no agent contains parallel code but that all
	//  agents are defined and collected. Then their behaviors and their configuration is 
	// analyzed and the agent simulator decides which behavior to call in which order
	//  so that the results can be fed into all agents. Or more concrete: each behavior is a kernel 
	// and the agent data is a stream that gets channeled through the connected kernels. Some 
	// calculations are unnecessary for some agents but the uniform calculations with very little 
	// ifs might speed up the whole enchilada. (On a streaming architecture at least -
	//  this might be very bad on normal multi-core CPUs)

	// Jens:
	// I would not expect any nested parallelism to gain speed, but it should be interesting to try it.
	// If you ask me, any try of using any kind a streamin approach on a "normal" cpu should at least
	// use the SIMD-extension availible on the cpu. :-)
		return where_I_want_to_be_next;
}


/// Ok, I have to admit the name "sensor" may be missleading, but I will keep it for now. :-)
/// This is the agents interface to any data it knows about the world
/// Using such a approach would inherant propose a lazy-evaluation idea. This may be of no use in open-steer
/// but in generic system it may lead to the result that we don't have to calculate e.g. the neighbours of an
/// agent in every frame without directly taken care of such a problem.


// bknafla:
// The idea of sensors for agents is gold - but do we really want to specify a whole sensor model just for the
// steering library? Do we really need a whole agent model for the steering? ;-)
// To decide this we have to analyze the way steering is actually used and how to ease its deployment (
// while allwing easy parallelization...).

// Jens:
// Well, for the steering library we need to specify at least how to access data. We could specify just a special
// datastructure ... or a generic interface to access the data. We may than call this interface "sensor"
// and here we are again. :-)
// Steering can not be done without any data and this has to be clearly specified; how we call this
// in the end will not change anything. Or how do you want to do this without specifying access to the data?

class sensor {
	vector<Vec3> neighbours();
	vector<Vec3> avoid();
}

vector<Vec3> sensor::neighbours() {
	//generate a agent specific neighbourhood
	//possible cache it, so this is not generated again again in the same frame
}

vector<Vec3> sensor::avoid() {
	//some neural network algorithm which calculates places to avoid
	//or just return a trivial precomputed list :-)
}
